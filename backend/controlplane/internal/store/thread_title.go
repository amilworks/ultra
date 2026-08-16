package store

import (
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const threadTitleStateKey = "title_state"

func mergeThreadMetadata(existing domain.JSONMap, incoming domain.JSONMap) domain.JSONMap {
	merged := copyJSONMap(existing)
	for key, value := range incoming {
		merged[key] = value
	}
	return merged
}

func copyJSONMap(value domain.JSONMap) domain.JSONMap {
	copied := domain.JSONMap{}
	for key, item := range value {
		copied[key] = item
	}
	return copied
}

func generatedThreadTitleEligible(thread domain.ThreadRecord) bool {
	title := normalizedThreadTitle(thread.Title)
	if title == "" || strings.EqualFold(title, "New conversation") {
		return true
	}
	source := threadTitleStateSource(thread.Metadata)
	switch source {
	case "manual", "generated":
		return false
	case "auto", "auto_initial", "initial_request":
		return true
	}
	bridge, _ := thread.Metadata["frontend_bridge"].(string)
	return strings.EqualFold(strings.TrimSpace(bridge), "v2-chat")
}

func generatedThreadTitleMetadata(
	existing domain.JSONMap,
	input domain.ApplyGeneratedThreadTitleInput,
	previousTitle string,
	now time.Time,
) domain.JSONMap {
	metadata := copyJSONMap(existing)
	state := domain.JSONMap{}
	for key, value := range input.Generation {
		state[key] = value
	}
	state["source"] = "generated"
	state["run_id"] = strings.TrimSpace(input.RunID)
	state["previous_title"] = strings.TrimSpace(previousTitle)
	state["updated_at"] = now.UTC().Format(time.RFC3339Nano)
	metadata[threadTitleStateKey] = state
	return metadata
}

func normalizedThreadTitle(value string) string {
	return strings.Join(strings.Fields(strings.Trim(value, " \t\r\n\"'`")), " ")
}

func threadVisibleForUserRead(thread domain.ThreadRecord) bool {
	return thread.Status != domain.ThreadStatusDeleted
}

func threadTitleStateSource(metadata domain.JSONMap) string {
	state, ok := metadata[threadTitleStateKey].(domain.JSONMap)
	if !ok {
		if generic, genericOK := metadata[threadTitleStateKey].(map[string]any); genericOK {
			state = domain.JSONMap(generic)
		}
	}
	if state == nil {
		return ""
	}
	source, _ := state["source"].(string)
	return strings.ToLower(strings.TrimSpace(source))
}
