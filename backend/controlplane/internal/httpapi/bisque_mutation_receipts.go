package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const bisqueMutationEventPageSize = 1000

type workerBisqueMutationReceipt struct {
	RunID       string
	ThreadID    string
	Action      string
	RequestSHA  string
	StartedID   string
	CompletedID string
	AmbiguousID string
}

type bisqueMutationUploadTarget struct {
	Kind      string `json:"kind"`
	Name      string `json:"name"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type bisqueUploadMutationRequest struct {
	Targets []bisqueMutationUploadTarget `json:"targets"`
}

type bisqueDatasetMutationRequest struct {
	Name         string   `json:"name"`
	ResourceURIs []string `json:"resource_uris"`
}

// beginWorkerBisqueMutation is a durable fail-closed idempotency barrier for
// autonomous external writes. A completed identical request replays its sealed
// response. A started-but-unsealed request is never blindly retried because an
// upstream timeout may have committed the side effect.
func (deps ServerDeps) beginWorkerBisqueMutation(
	ctx context.Context,
	authority bisqueRequestAuthority,
	action string,
	request any,
) (workerBisqueMutationReceipt, domain.JSONMap, bool, error) {
	if !authority.Worker {
		return workerBisqueMutationReceipt{}, nil, false, nil
	}
	requestMap, err := bisqueMutationJSONMap(request)
	if err != nil {
		return workerBisqueMutationReceipt{}, nil, false, err
	}
	canonical, err := json.Marshal(requestMap)
	if err != nil {
		return workerBisqueMutationReceipt{}, nil, false, err
	}
	digest := sha256.Sum256(append([]byte(action+"\x00"), canonical...))
	requestSHA := hex.EncodeToString(digest[:])
	baseID := "evt_" + authority.Run.RunID + "_bisque_" + requestSHA[:24]
	receipt := workerBisqueMutationReceipt{
		RunID:       authority.Run.RunID,
		ThreadID:    authority.Run.ThreadID,
		Action:      action,
		RequestSHA:  requestSHA,
		StartedID:   baseID + "_started",
		CompletedID: baseID + "_completed",
		AmbiguousID: baseID + "_ambiguous",
	}
	if cached, found, err := deps.completedWorkerBisqueMutation(ctx, receipt); err != nil {
		return receipt, nil, false, err
	} else if found {
		return receipt, cached, true, nil
	}
	if _, found, err := deps.Store.GetRunEvent(ctx, receipt.StartedID); err != nil {
		return receipt, nil, false, err
	} else if found {
		return receipt, nil, false, fmt.Errorf("%w: prior BisQue mutation outcome requires reconciliation", store.ErrConflict)
	}
	_, err = deps.Store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   receipt.StartedID,
		RunID:     receipt.RunID,
		ThreadID:  receipt.ThreadID,
		EventKind: "remote_mutation.started",
		EventType: "bisque",
		Level:     "info",
		Message:   "Run-bound BisQue mutation started.",
		Payload: domain.JSONMap{
			"action":         action,
			"request_sha256": requestSHA,
			"request":        requestMap,
		},
	})
	if err == nil {
		return receipt, nil, false, nil
	}
	if !errors.Is(err, store.ErrConflict) {
		return receipt, nil, false, err
	}
	if cached, found, getErr := deps.completedWorkerBisqueMutation(ctx, receipt); getErr != nil {
		return receipt, nil, false, getErr
	} else if found {
		return receipt, cached, true, nil
	}
	return receipt, nil, false, fmt.Errorf("%w: concurrent BisQue mutation is already in progress", store.ErrConflict)
}

func (deps ServerDeps) completeWorkerBisqueMutation(
	ctx context.Context,
	receipt workerBisqueMutationReceipt,
	response any,
) error {
	if receipt.RunID == "" {
		return nil
	}
	responseMap, err := bisqueMutationJSONMap(response)
	if err != nil {
		return err
	}
	_, err = deps.Store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   receipt.CompletedID,
		RunID:     receipt.RunID,
		ThreadID:  receipt.ThreadID,
		EventKind: "remote_mutation.completed",
		EventType: "bisque",
		Level:     "info",
		Message:   "Run-bound BisQue mutation completed.",
		Payload: domain.JSONMap{
			"action":         receipt.Action,
			"request_sha256": receipt.RequestSHA,
			"response":       responseMap,
		},
	})
	if errors.Is(err, store.ErrConflict) {
		return nil
	}
	return err
}

func (deps ServerDeps) markWorkerBisqueMutationAmbiguous(
	ctx context.Context,
	receipt workerBisqueMutationReceipt,
	err error,
) {
	if receipt.RunID == "" {
		return
	}
	payload := domain.JSONMap{
		"action":                  receipt.Action,
		"request_sha256":          receipt.RequestSHA,
		"reconciliation_required": true,
		"failure_class":           fmt.Sprintf("%T", err),
	}
	_, _ = deps.Store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   receipt.AmbiguousID,
		RunID:     receipt.RunID,
		ThreadID:  receipt.ThreadID,
		EventKind: "remote_mutation.ambiguous",
		EventType: "bisque",
		Level:     "error",
		Message:   "BisQue mutation outcome requires reconciliation.",
		Payload:   payload,
	})
}

func (deps ServerDeps) completedWorkerBisqueMutation(
	ctx context.Context,
	receipt workerBisqueMutationReceipt,
) (domain.JSONMap, bool, error) {
	event, found, err := deps.Store.GetRunEvent(ctx, receipt.CompletedID)
	if err != nil || !found {
		return nil, false, err
	}
	response, ok := jsonMapValue(event.Payload["response"])
	if !ok {
		return nil, false, fmt.Errorf("sealed BisQue mutation response is malformed")
	}
	return response, true, nil
}

func bisqueMutationJSONMap(value any) (domain.JSONMap, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var decoded domain.JSONMap
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		return nil, err
	}
	if decoded == nil {
		return nil, errors.New("BisQue mutation request must be a JSON object")
	}
	return decoded, nil
}

func (deps ServerDeps) workerBisqueUploadedResourceURIs(ctx context.Context, runID string) (map[string]bool, error) {
	allowed := map[string]bool{}
	cursor := int64(0)
	for {
		events, err := deps.Store.ListRunEventsAfter(ctx, runID, cursor, bisqueMutationEventPageSize)
		if err != nil {
			return nil, err
		}
		for _, event := range events {
			if event.Sequence > cursor {
				cursor = event.Sequence
			}
			if event.EventKind != "remote_mutation.completed" || event.Payload["action"] != "bisque.upload" {
				continue
			}
			response, ok := jsonMapValue(event.Payload["response"])
			if !ok {
				continue
			}
			for _, upload := range metadataJSONMaps(response["uploads"]) {
				uri, _ := safeMetadataString(upload["resource_uri"], 4096)
				if uri != "" {
					allowed[uri] = true
				}
			}
		}
		if len(events) < bisqueMutationEventPageSize {
			break
		}
	}
	return allowed, nil
}

func workerRunAllowsBisqueDatasetURI(run domain.RunRecord, uri string, uploaded map[string]bool) bool {
	uri = strings.TrimSpace(uri)
	if uri == "" {
		return false
	}
	if uploaded[uri] {
		return true
	}
	for _, selected := range metadataStringValues(run.Metadata["resource_uris"]) {
		if strings.TrimSpace(selected) == uri {
			return true
		}
	}
	return false
}

func canonicalBisqueDatasetMembers(values []string) []string {
	values = uniqueTrimmedStringValues(values)
	sort.Strings(values)
	return values
}

func verifiedBisqueMutationTarget(kind string, name string, path string, expectedSHA string, expectedSize int64) (bisqueMutationUploadTarget, []byte, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return bisqueMutationUploadTarget{}, nil, store.ErrNotFound
	}
	if expectedSize > 0 && int64(len(content)) != expectedSize {
		return bisqueMutationUploadTarget{}, nil, fmt.Errorf("%w: BisQue upload source size changed", store.ErrConflict)
	}
	digestBytes := sha256.Sum256(content)
	digest := hex.EncodeToString(digestBytes[:])
	expectedSHA = strings.ToLower(strings.TrimSpace(expectedSHA))
	if expectedSHA != "" && subtleDigestMismatch(expectedSHA, digest) {
		return bisqueMutationUploadTarget{}, nil, fmt.Errorf("%w: BisQue upload source digest changed", store.ErrConflict)
	}
	return bisqueMutationUploadTarget{
		Kind:      kind,
		Name:      safeOriginalFilename(name),
		SHA256:    digest,
		SizeBytes: int64(len(content)),
	}, content, nil
}

func canonicalBisqueUploadTargets(values []bisqueMutationUploadTarget) []bisqueMutationUploadTarget {
	values = append([]bisqueMutationUploadTarget(nil), values...)
	sort.Slice(values, func(i, j int) bool {
		left := values[i].Kind + "\x00" + values[i].SHA256 + "\x00" + values[i].Name
		right := values[j].Kind + "\x00" + values[j].SHA256 + "\x00" + values[j].Name
		if left == right {
			return values[i].SizeBytes < values[j].SizeBytes
		}
		return left < right
	})
	return values
}

func subtleDigestMismatch(expected string, actual string) bool {
	if len(expected) != len(actual) {
		return true
	}
	var mismatch byte
	for index := range expected {
		mismatch |= expected[index] ^ actual[index]
	}
	return mismatch != 0
}
