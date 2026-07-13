package domain

import "strings"

// RemoteMutationIntent is a server-validated capability for one family of
// external side effects. It is deliberately narrower than a tool name: a run
// may read/search BisQue without receiving either mutation capability.
type RemoteMutationIntent string

const (
	RemoteMutationIntentBisqueUpload        RemoteMutationIntent = "bisque.upload"
	RemoteMutationIntentBisqueCreateDataset RemoteMutationIntent = "bisque.create_dataset"

	// RemoteMutationIntentsMetadataKey is reserved control-plane metadata. HTTP
	// callers may request intents through the typed create-run field, never by
	// writing this key into the free-form metadata bag.
	RemoteMutationIntentsMetadataKey = "remote_mutation_intents"
	// BisqueAccountBindingMetadataKey holds a non-secret digest of the linked
	// destination selected at run creation. Workers never get authority merely
	// because the same user links a different account later.
	BisqueAccountBindingMetadataKey = "bisque_account_binding"
)

var remoteMutationIntentOrder = []RemoteMutationIntent{
	RemoteMutationIntentBisqueUpload,
	RemoteMutationIntentBisqueCreateDataset,
}

// ParseRemoteMutationIntents validates, de-duplicates, and canonicalizes a
// create-run request. Unknown values fail the whole request closed.
func ParseRemoteMutationIntents(values []string) ([]RemoteMutationIntent, bool) {
	wanted := make(map[RemoteMutationIntent]bool, len(values))
	for _, raw := range values {
		intent := RemoteMutationIntent(strings.ToLower(strings.TrimSpace(raw)))
		switch intent {
		case RemoteMutationIntentBisqueUpload, RemoteMutationIntentBisqueCreateDataset:
			wanted[intent] = true
		default:
			return nil, false
		}
	}
	parsed := make([]RemoteMutationIntent, 0, len(wanted))
	for _, intent := range remoteMutationIntentOrder {
		if wanted[intent] {
			parsed = append(parsed, intent)
		}
	}
	return parsed, true
}

func RemoteMutationIntentStrings(values []RemoteMutationIntent) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, string(value))
	}
	return out
}

// RemoteMutationIntentsFromMetadata accepts the concrete shapes produced by
// Go and PostgreSQL JSON decoding, then reuses the same strict enum parser.
func RemoteMutationIntentsFromMetadata(metadata JSONMap) ([]RemoteMutationIntent, bool) {
	if len(metadata) == 0 {
		return nil, true
	}
	raw, found := metadata[RemoteMutationIntentsMetadataKey]
	if !found || raw == nil {
		return nil, true
	}
	var values []string
	switch typed := raw.(type) {
	case []string:
		values = append(values, typed...)
	case []RemoteMutationIntent:
		for _, value := range typed {
			values = append(values, string(value))
		}
	case []any:
		for _, value := range typed {
			text, ok := value.(string)
			if !ok {
				return nil, false
			}
			values = append(values, text)
		}
	default:
		return nil, false
	}
	return ParseRemoteMutationIntents(values)
}

func HasRemoteMutationIntent(metadata JSONMap, required RemoteMutationIntent) bool {
	values, valid := RemoteMutationIntentsFromMetadata(metadata)
	if !valid {
		return false
	}
	for _, value := range values {
		if value == required {
			return true
		}
	}
	return false
}
