package store

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func resourceIDs(resources []domain.ResourceRecord) []string {
	ids := make([]string, 0, len(resources))
	for _, resource := range resources {
		ids = append(ids, resource.ResourceID)
	}
	return ids
}

func TestMemoryStoreRunEventPayloadIsImmutableAcrossBoundaries(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	type namedStringSlice []string
	type namedStringMap map[string]string
	type cyclicPayloadMap map[string]any
	type definedPointerTarget struct {
		Value  string
		Values []string
	}
	type definedPointer *definedPointerTarget
	type definedPointerHolder struct {
		Pointer definedPointer
	}
	type pointerCycle struct {
		Next *pointerCycle
	}
	type payloadStruct struct {
		Values   []string
		Metadata map[string]string
	}
	type payloadWithHiddenReference struct {
		hidden []string
	}

	newStoreAndRun := func(t *testing.T) (*MemoryStore, domain.RunRecord) {
		t.Helper()
		store := NewMemoryStore()
		thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
			UserID: "payload-owner",
			Title:  "Run-event payload isolation",
		})
		if err != nil {
			t.Fatalf("CreateThread: %v", err)
		}
		run, err := store.CreateRun(ctx, domain.CreateRunInput{
			ThreadID: thread.ThreadID,
			UserID:   "payload-owner",
			Goal:     "Verify run-event payload isolation",
		})
		if err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		return store, run
	}
	payloadFor := func(value string) domain.JSONMap {
		var nilSlice []string
		var nilMap map[string]string
		var nilNamedMap namedStringMap
		var nilRawMessage json.RawMessage
		var nilPointer *payloadStruct
		return domain.JSONMap{
			"domain_map": domain.JSONMap{
				"plain_map": map[string]any{
					"slice": []any{domain.JSONMap{"value": value}},
				},
			},
			"slice":            []any{map[string]any{"value": value}},
			"number":           float64(42.5),
			"boolean":          true,
			"null":             nil,
			"typed_slice":      []string{value},
			"named_slice":      namedStringSlice{value},
			"typed_map":        map[string]string{"value": value},
			"named_map":        namedStringMap{"value": value},
			"domain_map_slice": []domain.JSONMap{{"value": value}},
			"raw_message":      json.RawMessage(`{"stable":true}`),
			"pointer": &payloadStruct{
				Values:   []string{value},
				Metadata: map[string]string{"value": value},
			},
			"array":           [1][]string{{value}},
			"nil_slice":       nilSlice,
			"nil_map":         nilMap,
			"nil_named_map":   nilNamedMap,
			"nil_raw_message": nilRawMessage,
			"nil_pointer":     nilPointer,
		}
	}
	mutatePayload := func(t *testing.T, payload domain.JSONMap) {
		t.Helper()
		payload["domain_map"].(domain.JSONMap)["plain_map"].(map[string]any)["slice"].([]any)[0].(domain.JSONMap)["value"] = "mutated"
		payload["slice"].([]any)[0].(map[string]any)["value"] = "mutated"
		payload["number"] = float64(-1)
		payload["typed_slice"].([]string)[0] = "mutated"
		payload["named_slice"].(namedStringSlice)[0] = "mutated"
		payload["typed_map"].(map[string]string)["value"] = "mutated"
		payload["named_map"].(namedStringMap)["value"] = "mutated"
		payload["domain_map_slice"].([]domain.JSONMap)[0]["value"] = "mutated"
		payload["raw_message"].(json.RawMessage)[0] = '['
		payload["pointer"].(*payloadStruct).Values[0] = "mutated"
		payload["pointer"].(*payloadStruct).Metadata["value"] = "mutated"
		payload["array"].([1][]string)[0][0] = "mutated"
	}
	appendEvent := func(t *testing.T, store *MemoryStore, run domain.RunRecord, eventID string, sourceSequence int64, value string) domain.RunEventRecord {
		t.Helper()
		event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:        eventID,
			SourceSequence: sourceSequence,
			RunID:          run.RunID,
			ThreadID:       run.ThreadID,
			EventKind:      "payload.test",
			Payload:        payloadFor(value),
		})
		if err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
		return event
	}
	assertStoredPayload := func(t *testing.T, store *MemoryStore, eventID string, value string) {
		t.Helper()
		event, ok, err := store.GetRunEvent(ctx, eventID)
		if err != nil {
			t.Fatalf("GetRunEvent: %v", err)
		}
		if !ok {
			t.Fatalf("GetRunEvent(%q) not found", eventID)
		}
		if want := payloadFor(value); !reflect.DeepEqual(event.Payload, want) {
			t.Fatalf("stored payload = %#v, want %#v", event.Payload, want)
		}
		if event.Payload["nil_slice"].([]string) != nil {
			t.Fatalf("nested typed nil slice = %#v, want nil", event.Payload["nil_slice"])
		}
		if event.Payload["nil_map"].(map[string]string) != nil {
			t.Fatalf("nested typed nil map = %#v, want nil", event.Payload["nil_map"])
		}
		if event.Payload["nil_named_map"].(namedStringMap) != nil {
			t.Fatalf("nested named nil map = %#v, want nil", event.Payload["nil_named_map"])
		}
		if event.Payload["nil_raw_message"].(json.RawMessage) != nil {
			t.Fatalf("nested nil raw message = %#v, want nil", event.Payload["nil_raw_message"])
		}
		if event.Payload["nil_pointer"].(*payloadStruct) != nil {
			t.Fatalf("nested typed nil pointer = %#v, want nil", event.Payload["nil_pointer"])
		}
		encoded, err := json.Marshal(event.Payload)
		if err != nil {
			t.Fatalf("json.Marshal payload: %v", err)
		}
		var encodedFields map[string]json.RawMessage
		if err := json.Unmarshal(encoded, &encodedFields); err != nil {
			t.Fatalf("json.Unmarshal encoded payload: %v", err)
		}
		for _, key := range []string{"nil_slice", "nil_map", "nil_named_map", "nil_raw_message", "nil_pointer"} {
			if got := string(encodedFields[key]); got != "null" {
				t.Fatalf("encoded %s = %s, want null", key, got)
			}
		}
	}
	findEvent := func(t *testing.T, events []domain.RunEventRecord, eventID string) domain.RunEventRecord {
		t.Helper()
		for _, event := range events {
			if event.EventID == eventID {
				return event
			}
		}
		t.Fatalf("event %q not found in page", eventID)
		return domain.RunEventRecord{}
	}

	t.Run("original input after append", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		payload := payloadFor("input")
		event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "event-input",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payload,
		})
		if err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
		mutatePayload(t, payload)
		assertStoredPayload(t, store, event.EventID, "input")
	})

	t.Run("AppendRunEvent return", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		event := appendEvent(t, store, run, "event-append", 0, "append")
		mutatePayload(t, event.Payload)
		assertStoredPayload(t, store, event.EventID, "append")
	})

	t.Run("AppendRunEventIfRunActive appended return", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		event, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-active",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payloadFor("active"),
		})
		if err != nil {
			t.Fatalf("AppendRunEventIfRunActive: %v", err)
		}
		if outcome != RunEventAppendOutcomeAppended {
			t.Fatalf("outcome = %q, want %q", outcome, RunEventAppendOutcomeAppended)
		}
		mutatePayload(t, event.Payload)
		assertStoredPayload(t, store, event.EventID, "active")
	})

	t.Run("original input after AppendRunEventIfRunActive", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		payload := payloadFor("active input")
		event, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-active-input",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payload,
		})
		if err != nil {
			t.Fatalf("AppendRunEventIfRunActive: %v", err)
		}
		if outcome != RunEventAppendOutcomeAppended {
			t.Fatalf("outcome = %q, want %q", outcome, RunEventAppendOutcomeAppended)
		}
		mutatePayload(t, payload)
		assertStoredPayload(t, store, event.EventID, "active input")
	})

	t.Run("AppendRunEventIfRunActive duplicate return", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		original, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-duplicate",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payloadFor("duplicate"),
		})
		if err != nil {
			t.Fatalf("AppendRunEventIfRunActive append: %v", err)
		}
		if outcome != RunEventAppendOutcomeAppended {
			t.Fatalf("append outcome = %q, want %q", outcome, RunEventAppendOutcomeAppended)
		}
		duplicate, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   original.EventID,
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payloadFor("ignored duplicate input"),
		})
		if err != nil {
			t.Fatalf("AppendRunEventIfRunActive duplicate: %v", err)
		}
		if outcome != RunEventAppendOutcomeDuplicate {
			t.Fatalf("duplicate outcome = %q, want %q", outcome, RunEventAppendOutcomeDuplicate)
		}
		mutatePayload(t, duplicate.Payload)
		assertStoredPayload(t, store, original.EventID, "duplicate")
	})

	t.Run("GetRunEvent", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		event := appendEvent(t, store, run, "event-get", 0, "get")
		got, ok, err := store.GetRunEvent(ctx, event.EventID)
		if err != nil {
			t.Fatalf("GetRunEvent: %v", err)
		}
		if !ok {
			t.Fatal("GetRunEvent not found")
		}
		mutatePayload(t, got.Payload)
		assertStoredPayload(t, store, event.EventID, "get")
	})

	t.Run("GetRunEventBySourceSequence", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		event := appendEvent(t, store, run, "event-source-sequence", 41, "source sequence")
		got, ok, err := store.GetRunEventBySourceSequence(ctx, run.RunID, 41)
		if err != nil {
			t.Fatalf("GetRunEventBySourceSequence: %v", err)
		}
		if !ok {
			t.Fatal("GetRunEventBySourceSequence not found")
		}
		mutatePayload(t, got.Payload)
		assertStoredPayload(t, store, event.EventID, "source sequence")
	})

	listCases := []struct {
		name string
		list func(*testing.T, *MemoryStore, domain.RunRecord) []domain.RunEventRecord
	}{
		{
			name: "ListRunEvents",
			list: func(t *testing.T, store *MemoryStore, run domain.RunRecord) []domain.RunEventRecord {
				t.Helper()
				events, err := store.ListRunEvents(ctx, run.RunID, 10)
				if err != nil {
					t.Fatalf("ListRunEvents: %v", err)
				}
				return events
			},
		},
		{
			name: "ListRunEventsForUser",
			list: func(t *testing.T, store *MemoryStore, run domain.RunRecord) []domain.RunEventRecord {
				t.Helper()
				events, err := store.ListRunEventsForUser(ctx, run.RunID, run.UserID, 10)
				if err != nil {
					t.Fatalf("ListRunEventsForUser: %v", err)
				}
				return events
			},
		},
		{
			name: "ListRunEventsAfter",
			list: func(t *testing.T, store *MemoryStore, run domain.RunRecord) []domain.RunEventRecord {
				t.Helper()
				events, err := store.ListRunEventsAfter(ctx, run.RunID, 0, 10)
				if err != nil {
					t.Fatalf("ListRunEventsAfter: %v", err)
				}
				return events
			},
		},
		{
			name: "ListRunEventsAfterForUser",
			list: func(t *testing.T, store *MemoryStore, run domain.RunRecord) []domain.RunEventRecord {
				t.Helper()
				events, err := store.ListRunEventsAfterForUser(ctx, run.RunID, run.UserID, 0, 10)
				if err != nil {
					t.Fatalf("ListRunEventsAfterForUser: %v", err)
				}
				return events
			},
		},
	}
	for _, tc := range listCases {
		t.Run(tc.name, func(t *testing.T) {
			store, run := newStoreAndRun(t)
			event := appendEvent(t, store, run, "event-"+tc.name, 0, tc.name)
			listed := findEvent(t, tc.list(t, store, run), event.EventID)
			mutatePayload(t, listed.Payload)
			assertStoredPayload(t, store, event.EventID, tc.name)
		})
	}

	t.Run("nil payload remains non-nil empty map", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "event-empty-payload",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
		})
		if err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
		if event.Payload == nil || len(event.Payload) != 0 {
			t.Fatalf("appended payload = %#v, want non-nil empty map", event.Payload)
		}
		stored, ok, err := store.GetRunEvent(ctx, event.EventID)
		if err != nil {
			t.Fatalf("GetRunEvent: %v", err)
		}
		if !ok {
			t.Fatal("GetRunEvent not found")
		}
		if stored.Payload == nil || len(stored.Payload) != 0 {
			t.Fatalf("stored payload = %#v, want non-nil empty map", stored.Payload)
		}
	})

	newDefinedPointer := func(value string) definedPointer {
		return definedPointer(&definedPointerTarget{Value: value, Values: []string{value}})
	}
	definedPointerPayload := func(value string) domain.JSONMap {
		return domain.JSONMap{
			"interface_pointer": any(newDefinedPointer(value)),
			"pointer_slice":     []definedPointer{newDefinedPointer(value)},
			"pointer_map":       map[string]definedPointer{"value": newDefinedPointer(value)},
			"pointer_struct":    definedPointerHolder{Pointer: newDefinedPointer(value)},
			"pointer_key_map":   map[definedPointer]string{newDefinedPointer(value): "value"},
		}
	}
	assertDefinedPointers := func(t *testing.T, payload domain.JSONMap, value string) {
		t.Helper()
		assertPointer := func(name string, pointer definedPointer) {
			t.Helper()
			if pointer == nil || pointer.Value != value || !reflect.DeepEqual(pointer.Values, []string{value}) {
				t.Fatalf("%s = %#v, want defined pointer containing %q", name, pointer, value)
			}
		}
		pointer, ok := payload["interface_pointer"].(definedPointer)
		if !ok {
			t.Fatalf("interface pointer dynamic type = %T, want definedPointer", payload["interface_pointer"])
		}
		assertPointer("interface pointer", pointer)
		assertPointer("typed slice pointer", payload["pointer_slice"].([]definedPointer)[0])
		assertPointer("typed map pointer", payload["pointer_map"].(map[string]definedPointer)["value"])
		assertPointer("struct pointer", payload["pointer_struct"].(definedPointerHolder).Pointer)
		keyMap := payload["pointer_key_map"].(map[definedPointer]string)
		if len(keyMap) != 1 {
			t.Fatalf("pointer key map = %#v, want one entry", keyMap)
		}
		for key, mapValue := range keyMap {
			if mapValue != "value" {
				t.Fatalf("pointer key map value = %q, want value", mapValue)
			}
			assertPointer("map key pointer", key)
		}
	}
	mutateDefinedPointers := func(payload domain.JSONMap) {
		mutate := func(pointer definedPointer) {
			pointer.Value = "mutated"
			pointer.Values[0] = "mutated"
		}
		mutate(payload["interface_pointer"].(definedPointer))
		mutate(payload["pointer_slice"].([]definedPointer)[0])
		mutate(payload["pointer_map"].(map[string]definedPointer)["value"])
		mutate(payload["pointer_struct"].(definedPointerHolder).Pointer)
		for key := range payload["pointer_key_map"].(map[definedPointer]string) {
			mutate(key)
		}
	}

	t.Run("defined pointers preserve dynamic type and isolation", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		payload := definedPointerPayload("defined")
		var event domain.RunEventRecord
		var appendErr error
		var panicValue any
		func() {
			defer func() {
				panicValue = recover()
			}()
			event, appendErr = store.AppendRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "event-defined-pointers",
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "payload.test",
				Payload:   payload,
			})
		}()
		if panicValue != nil {
			t.Fatalf("AppendRunEvent panicked cloning defined pointer: %v", panicValue)
		}
		if appendErr != nil {
			t.Fatalf("AppendRunEvent: %v", appendErr)
		}
		assertDefinedPointers(t, event.Payload, "defined")
		mutateDefinedPointers(payload)
		assertDefinedPointers(t, event.Payload, "defined")
		mutateDefinedPointers(event.Payload)
		stored, ok, err := store.GetRunEvent(ctx, event.EventID)
		if err != nil || !ok {
			t.Fatalf("GetRunEvent = (_, %v, %v), want record", ok, err)
		}
		assertDefinedPointers(t, stored.Payload, "defined")
	})

	t.Run("finite overlapping slice views clone without a false cycle", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		outer := make([]any, 2)
		outer[0] = "x"
		outer[1] = outer[:1]
		event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "event-overlapping-slice",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   domain.JSONMap{"overlap": outer},
		})
		if err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
		assertOverlap := func(t *testing.T, payload domain.JSONMap) {
			t.Helper()
			encoded, err := json.Marshal(payload["overlap"])
			if err != nil {
				t.Fatalf("json.Marshal overlap: %v", err)
			}
			if got := string(encoded); got != `["x",["x"]]` {
				t.Fatalf("overlap JSON = %s, want [\"x\",[\"x\"]]", got)
			}
		}
		assertOverlap(t, event.Payload)
		outer[0] = "mutated input"
		assertOverlap(t, event.Payload)
		returned := event.Payload["overlap"].([]any)
		returned[0] = "mutated return"
		returned[1].([]any)[0] = "mutated nested return"
		stored, ok, err := store.GetRunEvent(ctx, event.EventID)
		if err != nil || !ok {
			t.Fatalf("GetRunEvent = (_, %v, %v), want record", ok, err)
		}
		assertOverlap(t, stored.Payload)
	})

	invalidPayloadCases := []struct {
		name    string
		payload func() domain.JSONMap
		want    error
	}{
		{
			name: "cycle",
			payload: func() domain.JSONMap {
				cycle := cyclicPayloadMap{}
				cycle["self"] = cycle
				return domain.JSONMap{"cycle": cycle}
			},
			want: errRunEventPayloadCycle,
		},
		{
			name: "self-referential slice",
			payload: func() domain.JSONMap {
				cycle := make([]any, 1)
				cycle[0] = cycle
				return domain.JSONMap{"cycle": cycle}
			},
			want: errRunEventPayloadCycle,
		},
		{
			name: "self-referential pointer",
			payload: func() domain.JSONMap {
				cycle := &pointerCycle{}
				cycle.Next = cycle
				return domain.JSONMap{"cycle": cycle}
			},
			want: errRunEventPayloadCycle,
		},
		{
			name: "channel",
			payload: func() domain.JSONMap {
				return domain.JSONMap{"channel": make(chan int)}
			},
			want: errRunEventPayloadUnsupported,
		},
		{
			name: "struct with hidden reference",
			payload: func() domain.JSONMap {
				return domain.JSONMap{"struct": payloadWithHiddenReference{hidden: []string{"secret"}}}
			},
			want: errRunEventPayloadUnsupported,
		},
	}
	for _, tc := range invalidPayloadCases {
		t.Run("AppendRunEvent rejects "+tc.name, func(t *testing.T) {
			store, run := newStoreAndRun(t)
			eventID := "event-invalid-append-" + tc.name
			if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   eventID,
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "payload.test",
				Payload:   tc.payload(),
			}); !errors.Is(err, tc.want) {
				t.Fatalf("AppendRunEvent error = %v, want %v", err, tc.want)
			}
			if _, ok, err := store.GetRunEvent(ctx, eventID); err != nil || ok {
				t.Fatalf("GetRunEvent after rejected append = (_, %v, %v), want (_, false, nil)", ok, err)
			}
		})

		t.Run("AppendRunEventIfRunActive rejects "+tc.name, func(t *testing.T) {
			store, run := newStoreAndRun(t)
			eventID := "event-invalid-active-" + tc.name
			_, _, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
				EventID:   eventID,
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "payload.test",
				Payload:   tc.payload(),
			})
			if !errors.Is(err, tc.want) {
				t.Fatalf("AppendRunEventIfRunActive error = %v, want %v", err, tc.want)
			}
			if _, ok, err := store.GetRunEvent(ctx, eventID); err != nil || ok {
				t.Fatalf("GetRunEvent after rejected active append = (_, %v, %v), want (_, false, nil)", ok, err)
			}
		})
	}

	t.Run("AppendRunEvent validates run existence before payload", func(t *testing.T) {
		store := NewMemoryStore()
		cycle := cyclicPayloadMap{}
		cycle["self"] = cycle
		if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "event-missing-run",
			RunID:     "missing-run",
			EventKind: "payload.test",
			Payload:   domain.JSONMap{"cycle": cycle},
		}); !errors.Is(err, ErrNotFound) {
			t.Fatalf("AppendRunEvent error = %v, want ErrNotFound", err)
		}
	})

	t.Run("AppendRunEventIfRunActive terminal duplicate ignores replacement payload", func(t *testing.T) {
		store, run := newStoreAndRun(t)
		original, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-duplicate-cycle",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   payloadFor("duplicate cycle"),
		})
		if err != nil || outcome != RunEventAppendOutcomeAppended {
			t.Fatalf("initial append = (_, %q, %v), want appended", outcome, err)
		}
		if _, err := store.CompleteRun(ctx, domain.CompleteRunInput{RunID: run.RunID}); err != nil {
			t.Fatalf("CompleteRun: %v", err)
		}
		cycle := cyclicPayloadMap{}
		cycle["self"] = cycle
		duplicate, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   original.EventID,
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   domain.JSONMap{"cycle": cycle},
		})
		if err != nil || outcome != RunEventAppendOutcomeDuplicate {
			t.Fatalf("duplicate replay = (_, %q, %v), want duplicate", outcome, err)
		}
		if !reflect.DeepEqual(duplicate.Payload, original.Payload) {
			t.Fatalf("duplicate payload = %#v, want %#v", duplicate.Payload, original.Payload)
		}
	})

	t.Run("AppendRunEventIfRunActive missing and terminal runs ignore payload", func(t *testing.T) {
		cyclePayload := func() domain.JSONMap {
			cycle := cyclicPayloadMap{}
			cycle["self"] = cycle
			return domain.JSONMap{"cycle": cycle}
		}
		store, run := newStoreAndRun(t)
		if _, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-missing-active-run",
			RunID:     "missing-run",
			EventKind: "payload.test",
			Payload:   cyclePayload(),
		}); err != nil || outcome != RunEventAppendOutcomeDropped {
			t.Fatalf("missing run append = (_, %q, %v), want dropped", outcome, err)
		}
		if _, err := store.CompleteRun(ctx, domain.CompleteRunInput{RunID: run.RunID}); err != nil {
			t.Fatalf("CompleteRun: %v", err)
		}
		if _, outcome, err := store.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
			EventID:   "event-terminal-active-run",
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "payload.test",
			Payload:   cyclePayload(),
		}); err != nil || outcome != RunEventAppendOutcomeDropped {
			t.Fatalf("terminal run append = (_, %q, %v), want dropped", outcome, err)
		}
	})
}

func TestMemoryStoreThreadRunEventArtifactFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Microscopy analysis",
		InitialMessages: []domain.ThreadMessage{{
			Role:    "user",
			Content: "Segment these images.",
		}},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	if thread.ThreadID == "" || thread.Status != domain.ThreadStatusActive {
		t.Fatalf("unexpected thread: %+v", thread)
	}

	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Segment these images.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Segment these images."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "Run started.",
		Payload:   domain.JSONMap{"phase": "planning"},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.EventID == "" {
		t.Fatalf("event id must be set")
	}

	artifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      run.RunID,
		ThreadID:   thread.ThreadID,
		Kind:       "report",
		Path:       "outputs/report.md",
		Title:      "Report",
		MimeType:   "text/markdown",
		SizeBytes:  42,
		SHA256:     "abc123",
		StorageURI: "file://outputs/report.md",
		Metadata:   domain.JSONMap{"source": "stub"},
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}
	if artifact.ArtifactID == "" {
		t.Fatalf("artifact id must be set")
	}

	events, err := store.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.started" {
		t.Fatalf("events = %+v, want one run.started", events)
	}

	artifacts, err := store.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/report.md" {
		t.Fatalf("artifacts = %+v, want report", artifacts)
	}
}

func TestMemoryStoreListThreadsPaginatesWithTotalCount(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	for _, title := range []string{"first", "second", "third", "fourth"} {
		if _, err := store.CreateThread(ctx, domain.CreateThreadInput{
			UserID: "user-1",
			Title:  title,
		}); err != nil {
			t.Fatalf("CreateThread %q: %v", title, err)
		}
		time.Sleep(time.Millisecond)
	}

	page, err := store.ListThreads(ctx, 2, 1, "")
	if err != nil {
		t.Fatalf("ListThreads: %v", err)
	}
	if page.TotalCount != 4 {
		t.Fatalf("total count = %d, want 4", page.TotalCount)
	}
	if page.Limit != 2 || page.Offset != 1 {
		t.Fatalf("page = limit %d offset %d, want limit 2 offset 1", page.Limit, page.Offset)
	}
	if len(page.Threads) != 2 {
		t.Fatalf("threads = %d, want 2", len(page.Threads))
	}
	if page.Threads[0].Title != "third" || page.Threads[1].Title != "second" {
		t.Fatalf("paged titles = %q, %q; want third, second", page.Threads[0].Title, page.Threads[1].Title)
	}
}

func TestMemoryStoreTenantScopedQueriesFilterByUser(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	aliceThread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          "alice",
		Title:           "Alice thread",
		InitialMessages: []domain.ThreadMessage{{Role: "user", Content: "alice message"}},
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}
	aliceRun, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: aliceThread.ThreadID,
		UserID:   "alice",
		Goal:     "alice run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "alice run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun alice: %v", err)
	}
	aliceEvent, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     aliceRun.RunID,
		ThreadID:  aliceThread.ThreadID,
		EventKind: "message.delta",
		Message:   "alice trace",
	})
	if err != nil {
		t.Fatalf("AppendRunEvent alice: %v", err)
	}
	aliceArtifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    aliceRun.RunID,
		ThreadID: aliceThread.ThreadID,
		Kind:     "report",
		Path:     "alice.md",
	})
	if err != nil {
		t.Fatalf("CreateArtifact alice: %v", err)
	}

	bobThread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "bob", Title: "Bob thread"})
	if err != nil {
		t.Fatalf("CreateThread bob: %v", err)
	}
	if _, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID,
		UserID:   "bob",
		Goal:     "bob run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bob run"}},
	}); err != nil {
		t.Fatalf("CreateRun bob: %v", err)
	}

	alicePage, err := store.ListThreadsForUser(ctx, "alice", 10, 0, "")
	if err != nil {
		t.Fatalf("ListThreadsForUser alice: %v", err)
	}
	if alicePage.TotalCount != 1 || len(alicePage.Threads) != 1 || alicePage.Threads[0].ThreadID != aliceThread.ThreadID {
		t.Fatalf("alice threads = %+v, want only alice thread", alicePage)
	}
	if _, err := store.GetThreadForUser(ctx, aliceThread.ThreadID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetThreadForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListThreadMessagesForUser(ctx, aliceThread.ThreadID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListThreadMessagesForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetRunForUser(ctx, aliceRun.RunID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRunForUser bob err = %v, want ErrNotFound", err)
	}
	bobRuns, err := store.ListRunsForUser(ctx, "bob", "", "", 10, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser bob: %v", err)
	}
	if len(bobRuns) != 1 || bobRuns[0].UserID != "bob" {
		t.Fatalf("bob runs = %+v, want only bob run", bobRuns)
	}
	if _, err := store.ListRunEventsForUser(ctx, aliceRun.RunID, "bob", 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunEventsAfterForUser(ctx, aliceRun.RunID, "bob", aliceEvent.Sequence-1, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsAfterForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunArtifactsForUser(ctx, aliceRun.RunID, "bob", 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunArtifactsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetArtifactForUser(ctx, aliceArtifact.ArtifactID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetArtifactForUser bob err = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreResourceCatalogFiltersSoftDeletesAndRestores(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	createdAt := time.Now().Add(-time.Hour).UTC()
	resource, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_alice",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		OriginalName: "cells.ome.tiff",
		ContentType:  "image/tiff",
		SizeBytes:    128,
		SHA256:       "abc123",
		StorageURI:   "file:///srv/ultra/shared/uploads/file_alice__cells.ome.tiff",
		StoragePath:  "file_alice__cells.ome.tiff",
		SourceType:   "upload",
		ResourceKind: "image",
		ProjectID:    "project-ct",
		Status:       "active",
		CreatedAt:    createdAt,
	})
	if err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	if resource.ResourceID != "file_alice" || resource.Status != "active" {
		t.Fatalf("resource = %+v, want active file_alice", resource)
	}
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_bob",
		OwnerUserID:  "bob",
		OwnerOrgID:   "org-a",
		OriginalName: "other.csv",
		SizeBytes:    64,
		SourceType:   "upload",
		ResourceKind: "table",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource bob: %v", err)
	}

	page, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:    "alice",
		OrgID:     "org-a",
		Query:     "ome",
		Kind:      "image",
		Source:    "upload",
		ProjectID: "project-ct",
		Limit:     20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser alice: %v", err)
	}
	if page.TotalCount != 1 || len(page.Resources) != 1 || page.Resources[0].ResourceID != "file_alice" {
		t.Fatalf("alice resources = %+v, want only alice image", page)
	}
	if page.Resources[0].ProjectID != "project-ct" {
		t.Fatalf("alice project_id = %q, want project-ct", page.Resources[0].ProjectID)
	}
	wrongProject, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice", OrgID: "org-a", ProjectID: "project-other", Limit: 20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser wrong project: %v", err)
	}
	if wrongProject.TotalCount != 0 || len(wrongProject.Resources) != 0 {
		t.Fatalf("wrong project resources = %+v, want none", wrongProject)
	}
	if _, err := store.GetResourceForUser(ctx, "file_alice", "bob", "org-a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser bob err = %v, want ErrNotFound", err)
	}

	deleted, err := store.SoftDeleteResourceForUser(ctx, "file_alice", "alice", "org-a", time.Now())
	if err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}
	if deleted.Status != "deleted" || deleted.DeletedAt.IsZero() || deleted.RetentionExpiresAt.IsZero() {
		t.Fatalf("deleted = %+v, want deleted status with retention expiry", deleted)
	}
	deletedPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{UserID: "alice", OrgID: "org-a", Limit: 20})
	if err != nil {
		t.Fatalf("ListResourcesForUser after delete: %v", err)
	}
	if deletedPage.TotalCount != 0 || len(deletedPage.Resources) != 0 {
		t.Fatalf("deleted resources = %+v, want no active rows", deletedPage)
	}
	stats, err := store.ResourceStorageStats(ctx)
	if err != nil {
		t.Fatalf("ResourceStorageStats: %v", err)
	}
	if stats.TotalResources != 1 || stats.TotalBytes != 64 {
		t.Fatalf("stats = %+v, want only bob active resource", stats)
	}

	restored, err := store.RestoreResourceForUser(ctx, "file_alice", "alice", "org-a", time.Now())
	if err != nil {
		t.Fatalf("RestoreResourceForUser: %v", err)
	}
	if restored.Status != "active" || !restored.DeletedAt.IsZero() || !restored.RetentionExpiresAt.IsZero() {
		t.Fatalf("restored = %+v, want active with empty retention fields", restored)
	}
	if _, err := store.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
		ResourceID:  "file_alice",
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		EventType:   "resource.restored",
	}); err != nil {
		t.Fatalf("CreateResourceEvent: %v", err)
	}
}

func TestMemoryStoreListResourceEventsForUserScopesAndFilters(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	base := time.Date(2026, 6, 8, 12, 0, 0, 0, time.UTC)
	inputs := []domain.UpsertResourceInput{
		{
			ResourceID:   "file_alice_active",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "alice-active.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base,
		},
		{
			ResourceID:   "file_alice_deleted",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "alice-deleted.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "deleted",
			CreatedAt:    base.Add(time.Minute),
			DeletedAt:    base.Add(5 * time.Minute),
		},
		{
			ResourceID:   "file_bob_private",
			OwnerUserID:  "bob",
			OwnerOrgID:   "org-b",
			OriginalName: "bob-private.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base.Add(2 * time.Minute),
		},
	}
	for _, input := range inputs {
		if _, err := store.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	if _, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID:      "file_alice_active",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "alice",
		CreatedAt:       base.Add(3 * time.Minute),
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	events := []domain.AppendResourceEventInput{
		{ResourceID: "file_alice_active", ActorUserID: "alice", ActorOrgID: "org-a", EventType: "resource.tagged", TS: base.Add(4 * time.Minute), Metadata: domain.JSONMap{"tag": "NPH"}},
		{ResourceID: "file_alice_deleted", ActorUserID: "alice", ActorOrgID: "org-a", EventType: "resource.deleted", TS: base.Add(5 * time.Minute)},
		{ResourceID: "file_bob_private", ActorUserID: "bob", ActorOrgID: "org-b", EventType: "resource.tagged", TS: base.Add(6 * time.Minute), Metadata: domain.JSONMap{"tag": "private"}},
	}
	for _, event := range events {
		if _, err := store.CreateResourceEvent(ctx, event); err != nil {
			t.Fatalf("CreateResourceEvent(%s): %v", event.ResourceID, err)
		}
	}

	aliceEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser alice: %v", err)
	}
	if aliceEvents.TotalCount != 2 || len(aliceEvents.Events) != 2 {
		t.Fatalf("alice events = %+v, want active and deleted owned events", aliceEvents)
	}
	if aliceEvents.Events[0].EventType != "resource.deleted" || aliceEvents.Events[1].EventType != "resource.tagged" {
		t.Fatalf("alice event order/types = %+v, want deleted then tagged", aliceEvents.Events)
	}

	deletedEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID:    "alice",
		OrgID:     "org-a",
		EventType: "resource.deleted",
		Limit:     10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser deleted: %v", err)
	}
	if deletedEvents.TotalCount != 1 || len(deletedEvents.Events) != 1 || deletedEvents.Events[0].ResourceID != "file_alice_deleted" {
		t.Fatalf("deleted events = %+v, want only alice deleted resource event", deletedEvents)
	}

	bobEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID:     "bob",
		OrgID:      "org-b",
		ResourceID: "file_alice_active",
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser bob shared resource: %v", err)
	}
	if bobEvents.TotalCount != 1 || len(bobEvents.Events) != 1 || bobEvents.Events[0].ResourceID != "file_alice_active" {
		t.Fatalf("bob shared events = %+v, want only shared active resource event", bobEvents)
	}

	foreignEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID: "charlie",
		OrgID:  "org-c",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser charlie: %v", err)
	}
	if foreignEvents.TotalCount != 0 || len(foreignEvents.Events) != 0 {
		t.Fatalf("charlie events = %+v, want no leaked audit events", foreignEvents)
	}
}

func TestMemoryStoreListResourceIDsForOwner(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_owner_org",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "owner-org.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
		{
			ResourceID:   "file_owner_no_org",
			OwnerUserID:  "alice",
			OriginalName: "owner-no-org.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
		{
			ResourceID:   "file_other_owner",
			OwnerUserID:  "bob",
			OwnerOrgID:   "org-a",
			OriginalName: "other-owner.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
	} {
		if _, err := store.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}

	existing, err := store.ListResourceIDsForOwner(ctx, " alice ", " org-a ", []string{
		" file_owner_org ",
		"file_owner_org",
		"file_owner_no_org",
		"file_other_owner",
		"file_missing",
		"",
	})
	if err != nil {
		t.Fatalf("ListResourceIDsForOwner: %v", err)
	}
	if !existing["file_owner_org"] || !existing["file_owner_no_org"] {
		t.Fatalf("existing = %+v, want owned org and no-org resources", existing)
	}
	if existing["file_other_owner"] || existing["file_missing"] || len(existing) != 2 {
		t.Fatalf("existing = %+v, want only resources visible to owner", existing)
	}
}

func TestMemoryStoreResourceCatalogSearchesDataAgentMetadata(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 15, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_captioned",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "image-001.png",
			ContentType:  "image/png",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata: domain.JSONMap{
				"label": "NPH",
				"data_agent": domain.JSONMap{
					"caption_resources": domain.JSONMap{
						"status":  "succeeded",
						"caption": "Prairie microscopy image with deterministic metadata caption.",
					},
				},
			},
		},
		{
			ResourceID:   "file_other",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "control-image.png",
			ContentType:  "image/png",
			SizeBytes:    64,
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"label": "Control"},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	captionMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Query:  "deterministic metadata caption",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser caption query: %v", err)
	}
	if captionMatches.TotalCount != 1 || len(captionMatches.Resources) != 1 || captionMatches.Resources[0].ResourceID != "file_captioned" {
		t.Fatalf("caption search resources = %+v, want captioned resource only", captionMatches)
	}

	labelMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Query:  "nph",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser metadata label query: %v", err)
	}
	if labelMatches.TotalCount != 1 || len(labelMatches.Resources) != 1 || labelMatches.Resources[0].ResourceID != "file_captioned" {
		t.Fatalf("metadata label search resources = %+v, want NPH resource only", labelMatches)
	}

	internalKeyMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Query:  "caption_resources",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser internal metadata key query: %v", err)
	}
	if internalKeyMatches.TotalCount != 0 || len(internalKeyMatches.Resources) != 0 {
		t.Fatalf("internal metadata key search resources = %+v, want no resources", internalKeyMatches)
	}
}

func TestMemoryStoreResourceCatalogFiltersByTags(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_nph_a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "patient-a.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Tags:         []string{"NPH", "Under 70", "MRI"},
		},
		{
			ResourceID:   "file_control",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "control.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Tags:         []string{"control", "MRI"},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Tags:   []string{"nph", "under 70"},
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser tag filter: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_nph_a" {
		t.Fatalf("tag-filtered resources = %+v, want only NPH under-70 resource", matches)
	}
	if got := matches.Resources[0].Tags; !reflect.DeepEqual(got, []string{"NPH", "Under 70", "MRI"}) {
		t.Fatalf("resource tags = %#v, want normalized persisted display tags", got)
	}

	queryMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Query:  "under 70",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser tag query: %v", err)
	}
	if queryMatches.TotalCount != 1 || len(queryMatches.Resources) != 1 || queryMatches.Resources[0].ResourceID != "file_nph_a" {
		t.Fatalf("tag query resources = %+v, want tag-matched NPH resource", queryMatches)
	}
}

func TestMemoryStoreResourceCatalogFiltersScientificMetadata(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_nph_under_70",
			OwnerUserID:  "metadata-user",
			OwnerOrgID:   "metadata-org",
			OriginalName: "sub-001.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(68),
			},
		},
		{
			ResourceID:   "file_nph_over_70",
			OwnerUserID:  "metadata-user",
			OwnerOrgID:   "metadata-org",
			OriginalName: "sub-002.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(74),
			},
		},
		{
			ResourceID:   "file_control_under_70",
			OwnerUserID:  "metadata-user",
			OwnerOrgID:   "metadata-org",
			OriginalName: "sub-003.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			Metadata: domain.JSONMap{
				"label":       "control",
				"format":      "nifti",
				"subject_age": float64(63),
			},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "metadata-user",
		OrgID:  "metadata-org",
		MetadataFilters: []domain.ResourceMetadataFilter{
			{Path: "label", Operator: "eq", Value: "NPH"},
			{Path: "format", Operator: "eq", Value: "nifti"},
			{Path: "subject_age", Operator: "lt", Value: "70"},
		},
		Limit: 20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser metadata filters: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_nph_under_70" {
		t.Fatalf("metadata-filtered resources = %+v, want only NPH under-70 NIfTI", matches)
	}
}

func TestMemoryStoreResourceSearchParsesScientificPredicatesAndFilePatterns(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 27, 9, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_old_64",
			OwnerUserID:  "search-user",
			OwnerOrgID:   "search-org",
			OriginalName: "Norm_old_004_64yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"label": "NPH",
				"image_header": domain.JSONMap{
					"reader":      "nifti-1",
					"array_dtype": "float32",
					"width":       float64(256),
				},
			},
		},
		{
			ResourceID:   "file_old_81",
			OwnerUserID:  "search-user",
			OwnerOrgID:   "search-org",
			OriginalName: "Norm_old_001_81yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"label": "control",
				"image_header": domain.JSONMap{
					"reader":      "nifti-1",
					"array_dtype": "uint16",
					"width":       float64(512),
				},
			},
		},
		{
			ResourceID:   "file_young_40",
			OwnerUserID:  "search-user",
			OwnerOrgID:   "search-org",
			OriginalName: "Norm_young_005_40yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
		{
			ResourceID:   "file_old_plain_72",
			OwnerUserID:  "search-user",
			OwnerOrgID:   "search-org",
			OriginalName: "Norm_old_008_72.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(-time.Second),
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
		{
			ResourceID:   "file_photo",
			OwnerUserID:  "search-user",
			OwnerOrgID:   "search-org",
			OriginalName: "prairie-camera.jpg",
			ContentType:  "image/jpeg",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    now.Add(3 * time.Second),
			Metadata: domain.JSONMap{
				"exif": domain.JSONMap{
					"camera_model":    "Sony A1",
					"focal_length_mm": float64(35),
					"iso":             float64(800),
				},
				"image_header": domain.JSONMap{
					"format": "jpeg",
					"width":  float64(2048),
					"height": float64(1024),
				},
			},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	ageMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser age query: %v", err)
	}
	if got := resourceIDs(ageMatches.Resources); !reflect.DeepEqual(got, []string{"file_old_81", "file_old_64", "file_old_plain_72"}) {
		t.Fatalf("age query resources = %v, want filename-derived old subjects", got)
	}

	combined, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "NPH age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser combined query: %v", err)
	}
	if got := resourceIDs(combined.Resources); !reflect.DeepEqual(got, []string{"file_old_64", "file_old_plain_72"}) {
		t.Fatalf("combined query resources = %v, want only NPH subject over 60", got)
	}

	nifti, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "*.nii.gz",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser glob query: %v", err)
	}
	if got := resourceIDs(nifti.Resources); !reflect.DeepEqual(got, []string{"file_young_40", "file_old_81", "file_old_64", "file_old_plain_72"}) {
		t.Fatalf("glob query resources = %v, want NIfTI gz resources", got)
	}

	niftiFamily, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "*.nii",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser NIfTI-family glob query: %v", err)
	}
	if got := resourceIDs(niftiFamily.Resources); !reflect.DeepEqual(got, []string{"file_young_40", "file_old_81", "file_old_64", "file_old_plain_72"}) {
		t.Fatalf("*.nii query resources = %v, want all NIfTI resources including .nii.gz", got)
	}

	headerMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "width > 1000",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser header query: %v", err)
	}
	if got := resourceIDs(headerMatches.Resources); !reflect.DeepEqual(got, []string{"file_photo"}) {
		t.Fatalf("header query resources = %v, want image header width match", got)
	}

	exifMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "search-user",
		OrgID:  "search-org",
		Query:  "focal_length > 30",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser exif query: %v", err)
	}
	if got := resourceIDs(exifMatches.Resources); !reflect.DeepEqual(got, []string{"file_photo"}) {
		t.Fatalf("EXIF query resources = %v, want focal-length match", got)
	}
}

func TestMemoryStoreResourceCatalogFiltersScientificDescriptors(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 9, 12, 30, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_nph_ventriculomegaly",
			OwnerUserID:  "descriptor-user",
			OwnerOrgID:   "descriptor-org",
			OriginalName: "sub-001.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"label":                  "NPH",
				"scientific_descriptors": []any{"ventriculomegaly", "MRI cohort"},
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{
						"status":      "succeeded",
						"descriptors": []any{"Evans index high", "lateral ventricle enlargement"},
					},
				},
			},
		},
		{
			ResourceID:   "file_control",
			OwnerUserID:  "descriptor-user",
			OwnerOrgID:   "descriptor-org",
			OriginalName: "sub-002.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"label":                  "control",
				"scientific_descriptors": []any{"normal ventricles", "MRI cohort"},
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{
						"status":      "succeeded",
						"descriptors": []any{"Evans index normal"},
					},
				},
			},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:      "descriptor-user",
		OrgID:       "descriptor-org",
		Descriptors: []string{"ventriculomegaly", "Evans index high"},
		Limit:       20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser descriptor filters: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_nph_ventriculomegaly" {
		t.Fatalf("descriptor-filtered resources = %+v, want only NPH ventriculomegaly resource", matches)
	}
}

func TestMemoryStoreResourceCatalogFiltersProcessingStatus(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 9, 13, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_caption_ready",
			OwnerUserID:  "processing-user",
			OwnerOrgID:   "processing-org",
			OriginalName: "caption-ready.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"data_agent": domain.JSONMap{
					"caption_resources": domain.JSONMap{"status": "succeeded"},
				},
			},
		},
		{
			ResourceID:   "file_metadata_ready",
			OwnerUserID:  "processing-user",
			OwnerOrgID:   "processing-org",
			OriginalName: "metadata-ready.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{"status": "succeeded"},
				},
			},
		},
		{
			ResourceID:   "file_needs_caption",
			OwnerUserID:  "processing-user",
			OwnerOrgID:   "processing-org",
			OriginalName: "needs-caption.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	captionReady, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:           "processing-user",
		OrgID:            "processing-org",
		ProcessingStatus: "caption_ready",
		Limit:            20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser caption_ready: %v", err)
	}
	if captionReady.TotalCount != 1 || len(captionReady.Resources) != 1 || captionReady.Resources[0].ResourceID != "file_caption_ready" {
		t.Fatalf("caption-ready resources = %+v, want only caption-ready resource", captionReady)
	}

	agentReady, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:           "processing-user",
		OrgID:            "processing-org",
		ProcessingStatus: "data_agent_ready",
		Limit:            20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser data_agent_ready: %v", err)
	}
	if agentReady.TotalCount != 2 {
		t.Fatalf("data-agent-ready resources = %+v, want caption or metadata ready resources", agentReady)
	}

	needsCaption, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:           "processing-user",
		OrgID:            "processing-org",
		ProcessingStatus: "needs_caption",
		Limit:            20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser needs_caption: %v", err)
	}
	if needsCaption.TotalCount != 2 {
		t.Fatalf("needs-caption resources = %+v, want resources without a successful caption job", needsCaption)
	}
}

func TestMemoryStoreResourceCatalogFiltersCreatedDate(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	base := time.Date(2026, 6, 1, 9, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_before_window",
			OwnerUserID:  "date-user",
			OwnerOrgID:   "date-org",
			OriginalName: "before-window.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    base,
		},
		{
			ResourceID:   "file_inside_window",
			OwnerUserID:  "date-user",
			OwnerOrgID:   "date-org",
			OriginalName: "inside-window.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    base.Add(48 * time.Hour),
		},
		{
			ResourceID:   "file_after_window",
			OwnerUserID:  "date-user",
			OwnerOrgID:   "date-org",
			OriginalName: "after-window.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    base.Add(96 * time.Hour),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:        "date-user",
		OrgID:         "date-org",
		CreatedAfter:  base.Add(24 * time.Hour),
		CreatedBefore: base.Add(72 * time.Hour),
		Limit:         20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser date filters: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_inside_window" {
		t.Fatalf("date-filtered resources = %+v, want only inside-window resource", matches)
	}
}

func TestMemoryStoreBulkTagResourcesForUserAuditsEachResource(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_batch_a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "batch-a.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			Tags:         []string{"raw"},
		},
		{
			ResourceID:   "file_batch_b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "batch-b.nii.gz",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	result, err := store.BulkTagResourcesForUser(ctx, domain.BulkTagResourcesInput{
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		ResourceIDs: []string{"file_batch_a", "file_batch_b"},
		Tags:        []string{"NPH", "MRI", "nph"},
	})
	if err != nil {
		t.Fatalf("BulkTagResourcesForUser: %v", err)
	}
	if result.UpdatedCount != 2 || len(result.Resources) != 2 || len(result.Events) != 2 {
		t.Fatalf("bulk tag result = %+v, want two resources and two audit events", result)
	}
	expectedTags := map[string][]string{
		"file_batch_a": []string{"raw", "NPH", "MRI"},
		"file_batch_b": []string{"NPH", "MRI"},
	}
	for _, resource := range result.Resources {
		if !reflect.DeepEqual(resource.Tags, expectedTags[resource.ResourceID]) {
			t.Fatalf("resource %s tags = %#v, want %#v", resource.ResourceID, resource.Tags, expectedTags[resource.ResourceID])
		}
		events, err := store.ListResourceEvents(ctx, resource.ResourceID, 10)
		if err != nil {
			t.Fatalf("ListResourceEvents(%s): %v", resource.ResourceID, err)
		}
		if len(events) != 1 || events[0].EventType != "resource.tagged" || events[0].Metadata["tags_added"] == nil {
			t.Fatalf("events for %s = %+v, want resource.tagged audit event", resource.ResourceID, events)
		}
	}
}

func TestMemoryStoreResourceReadGrantMakesResourceVisibleToGrantee(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 0, 0, 0, time.UTC)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_shared_read",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		OriginalName: "shared-nph-study.nii.gz",
		ContentType:  "application/gzip",
		SizeBytes:    512,
		SHA256:       "sha-shared-read",
		SourceType:   "upload",
		ResourceKind: "file",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}

	before, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser before grant: %v", err)
	}
	if before.TotalCount != 0 || len(before.Resources) != 0 {
		t.Fatalf("bob resources before grant = %+v, want none", before)
	}
	if _, err := store.GetResourceForUser(ctx, "file_shared_read", "bob", "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser before grant err = %v, want ErrNotFound", err)
	}

	grant, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		GrantID:         "resource_grant_shared_read",
		ResourceID:      "file_shared_read",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
		Metadata:        domain.JSONMap{"reason": "collaborative review"},
	})
	if err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if grant.Status != "active" || grant.ResourceID != "file_shared_read" || grant.GranteeUserID != "bob" || grant.Role != "read" {
		t.Fatalf("grant = %+v, want active bob read grant", grant)
	}

	after, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Query:  "shared-nph",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser after grant: %v", err)
	}
	if after.TotalCount != 1 || len(after.Resources) != 1 || after.Resources[0].ResourceID != "file_shared_read" {
		t.Fatalf("bob resources after grant = %+v, want shared resource", after)
	}
	if after.Resources[0].OwnerUserID != "alice" {
		t.Fatalf("shared resource owner = %q, want alice", after.Resources[0].OwnerUserID)
	}
	loaded, err := store.GetResourceForUser(ctx, "file_shared_read", "bob", "org-b")
	if err != nil {
		t.Fatalf("GetResourceForUser after grant: %v", err)
	}
	if loaded.ResourceID != "file_shared_read" || loaded.OwnerUserID != "alice" {
		t.Fatalf("loaded shared resource = %+v, want Alice-owned file", loaded)
	}
}

func TestMemoryStoreResourceShareGrantRevocationRemovesCollaboratorAccess(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 17, 0, 0, 0, time.UTC)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_revoked_share",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		OriginalName: "revoked-nph-study.nii.gz",
		ContentType:  "application/gzip",
		SizeBytes:    512,
		SHA256:       "sha-revoked-share",
		SourceType:   "upload",
		ResourceKind: "file",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	grant, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		GrantID:         "resource_grant_revoked_share",
		ResourceID:      "file_revoked_share",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if _, err := store.GetResourceForUser(ctx, "file_revoked_share", "bob", "org-b"); err != nil {
		t.Fatalf("GetResourceForUser before revoke: %v", err)
	}
	grants, err := store.ListResourceShareGrantsForResource(ctx, domain.ListResourceShareGrantsInput{
		ResourceID:  "file_revoked_share",
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("ListResourceShareGrantsForResource: %v", err)
	}
	if len(grants) != 1 || grants[0].GrantID != grant.GrantID || grants[0].Status != "active" {
		t.Fatalf("grants before revoke = %+v, want active grant", grants)
	}

	revokedAt := now.Add(2 * time.Second)
	revoked, err := store.RevokeResourceShareGrant(ctx, domain.RevokeResourceShareGrantInput{
		ResourceID:  "file_revoked_share",
		GrantID:     grant.GrantID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		RevokedAt:   revokedAt,
	})
	if err != nil {
		t.Fatalf("RevokeResourceShareGrant: %v", err)
	}
	if revoked.Status != "revoked" || !revoked.RevokedAt.Equal(revokedAt) {
		t.Fatalf("revoked grant = %+v, want revoked at %s", revoked, revokedAt)
	}
	if _, err := store.GetResourceForUser(ctx, "file_revoked_share", "bob", "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser after revoke err = %v, want ErrNotFound", err)
	}
	after, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Query:  "revoked-nph",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser after revoke: %v", err)
	}
	if after.TotalCount != 0 || len(after.Resources) != 0 {
		t.Fatalf("bob resources after revoke = %+v, want none", after)
	}
	allGrants, err := store.ListResourceShareGrantsForResource(ctx, domain.ListResourceShareGrantsInput{
		ResourceID:  "file_revoked_share",
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("ListResourceShareGrantsForResource after revoke: %v", err)
	}
	if len(allGrants) != 1 || allGrants[0].Status != "revoked" {
		t.Fatalf("grants after revoke = %+v, want revoked grant retained for audit", allGrants)
	}
}

func TestMemoryStoreUploadSessionLifecycle(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := domain.Now()

	session, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:          "upload_session_test",
		OwnerUserID:        "alice",
		OwnerOrgID:         "org-a",
		ProjectID:          "project-a",
		Status:             "active",
		TotalBytes:         12,
		IdempotencyKey:     "idem-1",
		BrowserFingerprint: "folder-a",
		CreatedAt:          now,
		UpdatedAt:          now,
	})
	if err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if session.SessionID != "upload_session_test" || session.Status != "active" {
		t.Fatalf("session = %+v, want active upload_session_test", session)
	}

	file, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:      session.SessionID,
		FileToken:      "file_token_a",
		OriginalName:   "cells.ome.tiff",
		RelativePath:   "batch-a/cells.ome.tiff",
		ContentType:    "image/tiff",
		SizeBytes:      12,
		Status:         "active",
		DeclaredSHA256: "declared",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("UpsertUploadSessionFile: %v", err)
	}
	if file.RelativePath != "batch-a/cells.ome.tiff" {
		t.Fatalf("file relative path = %q", file.RelativePath)
	}

	chunk, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  session.SessionID,
		FileToken:  file.FileToken,
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  12,
		SHA256:     "chunk-sha",
		Status:     "verified",
		StorageURI: "file://chunk",
		ReceivedAt: now,
		VerifiedAt: now,
	})
	if err != nil {
		t.Fatalf("UpsertUploadChunk: %v", err)
	}
	if chunk.Status != "verified" {
		t.Fatalf("chunk status = %q, want verified", chunk.Status)
	}

	loaded, err := store.GetUploadSessionForUser(ctx, session.SessionID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetUploadSessionForUser: %v", err)
	}
	if loaded.TotalBytes != 12 {
		t.Fatalf("loaded total bytes = %d, want 12", loaded.TotalBytes)
	}

	chunks, err := store.ListUploadSessionChunks(ctx, session.SessionID)
	if err != nil {
		t.Fatalf("ListUploadSessionChunks: %v", err)
	}
	if len(chunks) != 1 || chunks[0].FileToken != file.FileToken || chunks[0].ChunkIndex != 0 {
		t.Fatalf("session chunks = %+v, want uploaded chunk ordered by file token/index", chunks)
	}
	totals, err := store.GetUploadSessionTotals(ctx, session.SessionID)
	if err != nil {
		t.Fatalf("GetUploadSessionTotals: %v", err)
	}
	if totals.BytesReceived != 12 || totals.BytesVerified != 12 || totals.BytesCommitted != 0 || totals.AllComplete {
		t.Fatalf("session totals = %+v, want received/verified bytes without committed completion", totals)
	}
}

func TestMemoryStoreUploadChunkDoesNotReplaceVerifiedBytes(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := domain.Now()

	if _, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:   "upload_session_verified_conflict",
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Status:      "active",
		TotalBytes:  6,
		CreatedAt:   now,
		UpdatedAt:   now,
	}); err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:    "upload_session_verified_conflict",
		FileToken:    "file-a",
		OriginalName: "cells.ome.tiff",
		SizeBytes:    6,
		Status:       "uploading",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile: %v", err)
	}

	verified, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  "upload_session_verified_conflict",
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  6,
		SHA256:     "abcdef",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	})
	if err != nil {
		t.Fatalf("UpsertUploadChunk verified: %v", err)
	}
	replayed, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  "upload_session_verified_conflict",
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  6,
		SHA256:     "ABCDEF",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	})
	if err != nil {
		t.Fatalf("idempotent UpsertUploadChunk replay: %v", err)
	}
	if replayed.SHA256 != "ABCDEF" {
		t.Fatalf("idempotent replay chunk = %+v, want replay accepted with same digest", replayed)
	}

	if _, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  "upload_session_verified_conflict",
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  6,
		SHA256:     "different",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting verified UpsertUploadChunk err = %v, want ErrConflict", err)
	}
	chunks, err := store.ListUploadChunks(ctx, "upload_session_verified_conflict", "file-a")
	if err != nil {
		t.Fatalf("ListUploadChunks: %v", err)
	}
	if len(chunks) != 1 || !sameVerifiedUploadChunk(chunks[0], verified) {
		t.Fatalf("chunks after conflict = %+v, want original verified manifest", chunks)
	}
}

func TestMemoryStoreResourceCollectionBulkMembership(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_alice_a",
			OriginalName: "alice-a.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_alice_b",
			OriginalName: "alice-b.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "file_bob_foreign",
			OriginalName: "bob-private.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    20,
			OwnerUserID:  "bob",
			OwnerOrgID:   "org-b",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	collection, err := store.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_alice_nph",
		OwnerUserID:    "alice",
		OwnerOrgID:     "org-a",
		ProjectID:      "nph-study",
		Name:           "NPH NIfTI Files",
		Description:    "Curated NPH brain imaging resources",
		CollectionType: "folder",
		CreatedAt:      now,
		UpdatedAt:      now,
		Metadata:       domain.JSONMap{"label": "NPH"},
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if collection.CollectionID != "collection_alice_nph" || collection.CollectionType != "folder" || collection.ResourceCount != 0 {
		t.Fatalf("collection = %+v, want empty folder collection", collection)
	}

	added, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_alice_a", "file_alice_b"},
		AddedByUserID: "alice",
		AddedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}
	if added.AddedCount != 2 || len(added.Memberships) != 2 {
		t.Fatalf("added = %+v, want two collection memberships", added)
	}

	page, err := store.ListResourceCollectionsForUser(ctx, domain.ResourceCollectionListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Type:   "folder",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceCollectionsForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Collections) != 1 || page.Collections[0].ResourceCount != 2 {
		t.Fatalf("collection page = %+v, want one folder with two resources", page)
	}

	memberPage, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       "alice",
		OrgID:        "org-a",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser: %v", err)
	}
	if memberPage.TotalCount != 2 || len(memberPage.Resources) != 2 {
		t.Fatalf("member resources = %+v, want two resources", memberPage)
	}
	if got := []string{memberPage.Resources[0].ResourceID, memberPage.Resources[1].ResourceID}; got[0] != "file_alice_a" || got[1] != "file_alice_b" {
		t.Fatalf("member resource order = %+v, want insertion order", got)
	}

	if _, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_bob_foreign"},
		AddedByUserID: "alice",
		AddedAt:       now.Add(3 * time.Second),
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("AddResourcesToCollection foreign resource error = %v, want ErrNotFound", err)
	}

	if _, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       "bob",
		OrgID:        "org-b",
		Limit:        10,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign collection list error = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreResourceCollectionShareGrantAppliesToFutureMembers(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 20, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_folder_acl_initial",
			OriginalName: "initial-folder-acl.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_folder_acl_future",
			OriginalName: "future-folder-acl.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	collection, err := store.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_folder_acl",
		OwnerUserID:    "alice",
		OwnerOrgID:     "org-a",
		Name:           "Inherited ACL folder",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_folder_acl_initial"},
		AddedByUserID: "alice",
		AddedAt:       now,
	}); err != nil {
		t.Fatalf("AddResourcesToCollection initial: %v", err)
	}

	shareResult, err := store.CreateResourceCollectionShareGrant(ctx, domain.CreateResourceCollectionShareGrantInput{
		GrantID:         "collection_grant_folder_acl",
		CollectionID:    collection.CollectionID,
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"reason": "folder review"},
	})
	if err != nil {
		t.Fatalf("CreateResourceCollectionShareGrant: %v", err)
	}
	if shareResult.Grant.CollectionID != collection.CollectionID || len(shareResult.ResourceGrants) != 1 {
		t.Fatalf("shareResult = %+v, want collection grant and initial resource grant", shareResult)
	}
	if shareResult.ResourceGrants[0].ResourceID != "file_folder_acl_initial" {
		t.Fatalf("initial resource grant = %+v, want initial folder member", shareResult.ResourceGrants[0])
	}

	added, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_folder_acl_future"},
		AddedByUserID: "alice",
		AddedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("AddResourcesToCollection future: %v", err)
	}
	if len(added.InheritedShareGrants) != 1 {
		t.Fatalf("future inherited grants = %+v, want one inherited resource grant", added.InheritedShareGrants)
	}
	futureGrant := added.InheritedShareGrants[0]
	if futureGrant.ResourceID != "file_folder_acl_future" || futureGrant.GranteeUserID != "bob" || futureGrant.GranteeOrgID != "org-b" || futureGrant.Role != "read" || futureGrant.Status != "active" {
		t.Fatalf("future inherited grant = %+v, want active Bob read grant", futureGrant)
	}
	if futureGrant.Metadata["collection_share_grant_id"] != "collection_grant_folder_acl" || futureGrant.Metadata["source"] != "resource_collection_share_inherited" {
		t.Fatalf("future inherited grant metadata = %+v, want collection-share provenance", futureGrant.Metadata)
	}

	bobCollections, err := store.ListResourceCollectionsForUser(ctx, domain.ResourceCollectionListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Query:  "Inherited ACL",
		Type:   "folder",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceCollectionsForUser as bob: %v", err)
	}
	if bobCollections.TotalCount != 1 || len(bobCollections.Collections) != 1 || bobCollections.Collections[0].CollectionID != collection.CollectionID {
		t.Fatalf("bob collections = %+v, want shared folder", bobCollections)
	}
	bobCollection, err := store.GetResourceCollectionForUser(ctx, collection.CollectionID, "bob", "org-b")
	if err != nil {
		t.Fatalf("GetResourceCollectionForUser as bob: %v", err)
	}
	if bobCollection.ResourceCount != 2 {
		t.Fatalf("bob collection = %+v, want two visible members", bobCollection)
	}
	bobMembers, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       "bob",
		OrgID:        "org-b",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser as bob: %v", err)
	}
	if bobMembers.TotalCount != 2 || len(bobMembers.Resources) != 2 {
		t.Fatalf("bob folder resources = %+v, want shared current and future folder members", bobMembers)
	}
	if !bobMembers.Resources[0].ShareSummary.SharedWithMe || !bobMembers.Resources[1].ShareSummary.SharedWithMe {
		t.Fatalf("bob folder share summaries = %+v, %+v, want shared_with_me", bobMembers.Resources[0].ShareSummary, bobMembers.Resources[1].ShareSummary)
	}

	bobPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Query:  "folder-acl",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser as bob: %v", err)
	}
	if bobPage.TotalCount != 2 || len(bobPage.Resources) != 2 {
		t.Fatalf("bob resources = %+v, want both current and future folder members", bobPage)
	}
}

func TestMemoryStoreResourceCollectionResourceFilters(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 12, 0, 0, 0, time.UTC)

	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_nph_under_70",
			OriginalName: "nph-under-70.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata: domain.JSONMap{
				"cohort": "NPH",
				"data_agent": domain.JSONMap{
					"caption_resources": domain.JSONMap{
						"caption": "NPH patient under seventy with shunt imaging metadata.",
						"status":  "succeeded",
					},
				},
			},
		},
		{
			ResourceID:   "file_nph_table",
			OriginalName: "nph-clinical-table.csv",
			ResourceKind: "table",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "file_control_under_70",
			OriginalName: "control-under-70.nii.gz",
			ResourceKind: "file",
			SourceType:   "bisque_import",
			SizeBytes:    20,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	collection, err := store.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_alice_nph_filters",
		OwnerUserID:    "alice",
		OwnerOrgID:     "org-a",
		Name:           "NPH filter review",
		CollectionType: "folder",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_nph_under_70", "file_nph_table", "file_control_under_70"},
		AddedByUserID: "alice",
		AddedAt:       now.Add(3 * time.Second),
	}); err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}

	page, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       "alice",
		OrgID:        "org-a",
		Query:        "nph",
		Kind:         "file",
		Source:       "upload",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Resources) != 1 || page.Resources[0].ResourceID != "file_nph_under_70" {
		t.Fatalf("filtered collection resources = %+v, want only uploaded NPH file resource", page)
	}

	metadataPage, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       "alice",
		OrgID:        "org-a",
		Query:        "shunt imaging metadata",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser metadata query: %v", err)
	}
	if metadataPage.TotalCount != 1 || len(metadataPage.Resources) != 1 || metadataPage.Resources[0].ResourceID != "file_nph_under_70" {
		t.Fatalf("metadata-filtered collection resources = %+v, want NPH shunt resource", metadataPage)
	}
}

func TestMemoryStoreDatasetSnapshotFreezesResourceManifest(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 13, 30, 0, 0, time.UTC)

	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_snapshot_a",
			OriginalName: "nph-a.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    12,
			SHA256:       "sha-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_snapshot_b",
			OriginalName: "nph-b.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    15,
			SHA256:       "sha-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	snapshot, entries, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_nph_v1",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "project-nph",
		Name:            "NPH training cohort v1",
		Description:     "Frozen manifest for NPH model training",
		ResourceIDs:     []string{"file_snapshot_a", "file_snapshot_b"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"source": "unit_test"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}
	if snapshot.ResourceCount != 2 || snapshot.TotalBytes != 27 || len(entries) != 2 {
		t.Fatalf("snapshot = %+v entries=%+v, want frozen two-resource manifest", snapshot, entries)
	}
	if entries[0].ResourceID != "file_snapshot_a" || entries[0].SHA256 != "sha-a" || entries[0].SizeBytes != 12 {
		t.Fatalf("first snapshot entry = %+v, want original file_snapshot_a manifest", entries[0])
	}

	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_snapshot_a",
		OriginalName: "nph-a-renamed.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    99,
		SHA256:       "sha-a-mutated",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now.Add(3 * time.Second),
	}); err != nil {
		t.Fatalf("mutate resource: %v", err)
	}

	loaded, loadedEntries, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser: %v", err)
	}
	if loaded.SnapshotID != snapshot.SnapshotID || len(loadedEntries) != 2 {
		t.Fatalf("loaded snapshot = %+v entries=%+v, want same frozen manifest", loaded, loadedEntries)
	}
	if loadedEntries[0].OriginalName != "nph-a.nii.gz" || loadedEntries[0].SHA256 != "sha-a" || loadedEntries[0].SizeBytes != 12 {
		t.Fatalf("loaded first entry = %+v, want immutable pre-mutation manifest", loadedEntries[0])
	}

	page, err := store.ListDatasetSnapshotsForUser(ctx, domain.DatasetSnapshotListInput{
		UserID:    "alice",
		OrgID:     "org-a",
		ProjectID: "project-nph",
		Limit:     10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotsForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Snapshots) != 1 || page.Snapshots[0].SnapshotID != snapshot.SnapshotID {
		t.Fatalf("snapshot page = %+v, want the visible dataset snapshot", page)
	}
	otherUserPage, err := store.ListDatasetSnapshotsForUser(ctx, domain.DatasetSnapshotListInput{
		UserID: "bob",
		OrgID:  "org-a",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotsForUser other user: %v", err)
	}
	if otherUserPage.TotalCount != 0 || len(otherUserPage.Snapshots) != 0 {
		t.Fatalf("other user page = %+v, want owner-isolated snapshots", otherUserPage)
	}
}

func TestMemoryStoreDatasetSnapshotShareGrantAllowsCollaboratorRead(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 14, 0, 0, 0, time.UTC)

	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_shared_snapshot_a",
			OriginalName: "shared-nph-a.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    32,
			SHA256:       "sha-shared-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_shared_snapshot_b",
			OriginalName: "shared-nph-b.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    64,
			SHA256:       "sha-shared-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	snapshot, entries, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_share_v1",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		Name:            "Shared NPH cohort",
		ResourceIDs:     []string{"file_shared_snapshot_a", "file_shared_snapshot_b"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"source": "unit_test"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("snapshot entries = %+v, want frozen two-resource manifest", entries)
	}

	if _, _, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, "bob", "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDatasetSnapshotForUser before grant err = %v, want ErrNotFound", err)
	}
	beforeGrant, err := store.ListDatasetSnapshotsForUser(ctx, domain.DatasetSnapshotListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotsForUser before grant: %v", err)
	}
	if beforeGrant.TotalCount != 0 || len(beforeGrant.Snapshots) != 0 {
		t.Fatalf("before grant page = %+v, want no visible snapshots", beforeGrant)
	}

	grant, err := store.CreateDatasetSnapshotShareGrant(ctx, domain.CreateDatasetSnapshotShareGrantInput{
		GrantID:         "dataset_snapshot_grant_bob",
		SnapshotID:      snapshot.SnapshotID,
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		Metadata:        domain.JSONMap{"reason": "review cohort"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshotShareGrant: %v", err)
	}
	if grant.Status != "active" || grant.Role != "read" || grant.SnapshotID != snapshot.SnapshotID {
		t.Fatalf("grant = %+v, want active read snapshot grant", grant)
	}

	loaded, loadedEntries, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, "bob", "org-b")
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser after grant: %v", err)
	}
	if loaded.SnapshotID != snapshot.SnapshotID || loaded.OwnerUserID != "alice" {
		t.Fatalf("loaded shared snapshot = %+v, want original owner-visible snapshot", loaded)
	}
	if len(loadedEntries) != 2 || loadedEntries[0].SHA256 != "sha-shared-a" {
		t.Fatalf("loaded shared entries = %+v, want frozen manifest", loadedEntries)
	}
	afterGrant, err := store.ListDatasetSnapshotsForUser(ctx, domain.DatasetSnapshotListInput{
		UserID: "bob",
		OrgID:  "org-b",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotsForUser after grant: %v", err)
	}
	if afterGrant.TotalCount != 1 || len(afterGrant.Snapshots) != 1 || afterGrant.Snapshots[0].SnapshotID != snapshot.SnapshotID {
		t.Fatalf("after grant page = %+v, want shared snapshot", afterGrant)
	}

	grants, err := store.ListDatasetSnapshotShareGrants(ctx, domain.ListDatasetSnapshotShareGrantsInput{
		SnapshotID:  snapshot.SnapshotID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Status:      "active",
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotShareGrants: %v", err)
	}
	if len(grants) != 1 || grants[0].GrantID != grant.GrantID {
		t.Fatalf("grants = %+v, want owner-visible active grant", grants)
	}

	revoked, err := store.RevokeDatasetSnapshotShareGrant(ctx, domain.RevokeDatasetSnapshotShareGrantInput{
		SnapshotID:       snapshot.SnapshotID,
		GrantID:          grant.GrantID,
		OwnerUserID:      "alice",
		OwnerOrgID:       "org-a",
		RevokedByUserID:  "alice",
		RevokedAt:        now.Add(4 * time.Second),
		RevocationReason: "cohort review complete",
	})
	if err != nil {
		t.Fatalf("RevokeDatasetSnapshotShareGrant: %v", err)
	}
	if revoked.Status != "revoked" || revoked.RevokedAt.IsZero() {
		t.Fatalf("revoked grant = %+v, want revoked lifecycle state", revoked)
	}
	if _, _, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, "bob", "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDatasetSnapshotForUser after revoke err = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreDatasetSnapshotEventsAuditCreateShareRevoke(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 15, 0, 0, 0, time.UTC)

	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_event_snapshot_a",
		OriginalName: "event-nph-a.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    48,
		SHA256:       "sha-event-a",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		ProjectID:    "project-event",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	snapshot, _, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_events_v1",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "project-event",
		Name:            "Evented NPH cohort",
		ResourceIDs:     []string{"file_event_snapshot_a"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
		Metadata:        domain.JSONMap{"source": "unit_test"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}

	ownerEvents, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     "alice",
		OrgID:      "org-a",
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser owner after create: %v", err)
	}
	if ownerEvents.TotalCount != 1 || len(ownerEvents.Events) != 1 || ownerEvents.Events[0].EventType != "dataset_snapshot.created" {
		t.Fatalf("owner events after create = %+v, want dataset_snapshot.created", ownerEvents)
	}
	if ownerEvents.Events[0].ActorUserID != "alice" || ownerEvents.Events[0].Metadata["snapshot_name"] != "Evented NPH cohort" {
		t.Fatalf("created event = %+v, want actor and snapshot metadata", ownerEvents.Events[0])
	}
	if _, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     "bob",
		OrgID:      "org-b",
		Limit:      10,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("bob events before grant err = %v, want ErrNotFound", err)
	}

	grant, err := store.CreateDatasetSnapshotShareGrant(ctx, domain.CreateDatasetSnapshotShareGrantInput{
		GrantID:         "dataset_snapshot_event_grant_bob",
		SnapshotID:      snapshot.SnapshotID,
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshotShareGrant: %v", err)
	}
	bobEvents, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     "bob",
		OrgID:      "org-b",
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser bob after grant: %v", err)
	}
	if bobEvents.TotalCount != 2 || len(bobEvents.Events) != 2 || bobEvents.Events[0].EventType != "dataset_snapshot.shared" || bobEvents.Events[0].Metadata["grant_id"] != grant.GrantID {
		t.Fatalf("bob events after grant = %+v, want shared then created events", bobEvents)
	}

	if _, err := store.RevokeDatasetSnapshotShareGrant(ctx, domain.RevokeDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshot.SnapshotID,
		GrantID:         grant.GrantID,
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		RevokedByUserID: "alice",
		RevokedAt:       now.Add(3 * time.Second),
	}); err != nil {
		t.Fatalf("RevokeDatasetSnapshotShareGrant: %v", err)
	}
	ownerEvents, err = store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     "alice",
		OrgID:      "org-a",
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser owner after revoke: %v", err)
	}
	if ownerEvents.TotalCount != 3 || len(ownerEvents.Events) != 3 || ownerEvents.Events[0].EventType != "dataset_snapshot.share_revoked" {
		t.Fatalf("owner events after revoke = %+v, want revoke/share/create audit trail", ownerEvents)
	}
	if _, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     "bob",
		OrgID:      "org-b",
		Limit:      10,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("bob events after revoke err = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreDatasetSnapshotFromResourceQueryFreezesMatches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_query_old_nph",
			OriginalName: "NPH_shunt_001_69yo.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    21,
			SHA256:       "sha-query-old",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "project-nph",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"age": 69, "diagnosis": "NPH"},
		},
		{
			ResourceID:   "file_query_young_nph",
			OriginalName: "NPH_shunt_002_62yo.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    34,
			SHA256:       "sha-query-young",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "project-nph",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"age": 62, "diagnosis": "NPH"},
		},
		{
			ResourceID:   "file_query_older_nph",
			OriginalName: "NPH_shunt_003_74yo.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    55,
			SHA256:       "sha-query-older",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "project-nph",
			Status:       "active",
			Tags:         []string{"NPH", "Over 70"},
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
			Metadata:     domain.JSONMap{"age": 74, "diagnosis": "NPH"},
		},
		{
			ResourceID:   "file_query_control_under70",
			OriginalName: "control_004_66yo.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    89,
			SHA256:       "sha-query-control",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "project-nph",
			Status:       "active",
			Tags:         []string{"Under 70"},
			CreatedAt:    now.Add(3 * time.Second),
			UpdatedAt:    now.Add(3 * time.Second),
			Metadata:     domain.JSONMap{"age": 66, "diagnosis": "control"},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	snapshot, entries, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_query_under70",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		Name:            "NPH under 70 query cohort",
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		ResourceQuery: &domain.DatasetSnapshotResourceQuery{
			Query:     "NPH",
			ProjectID: "project-nph",
			Kind:      "file",
			Source:    "upload",
			Tags:      []string{"Under 70"},
			MetadataFilters: []domain.ResourceMetadataFilter{
				{Path: "diagnosis", Operator: "eq", Value: "NPH"},
				{Path: "age", Operator: "lt", Value: "70"},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot query: %v", err)
	}
	if snapshot.ResourceCount != 2 || snapshot.TotalBytes != 55 || len(entries) != 2 {
		t.Fatalf("query snapshot = %+v entries=%+v, want two matching under-70 files", snapshot, entries)
	}
	got := []string{entries[0].ResourceID, entries[1].ResourceID}
	if got[0] != "file_query_young_nph" || got[1] != "file_query_old_nph" {
		t.Fatalf("query snapshot order = %v, want newest matching resources first", got)
	}
}

func TestMemoryStoreDataAgentJobLifecycleRecordsEvents(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 14, 30, 0, 0, time.UTC)

	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_agent_a",
			OriginalName: "nph-a.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_agent_b",
			OriginalName: "nph-b.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	job, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_nph_caption",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "caption_resources",
		ResourceIDs:     []string{"file_agent_a", "file_agent_b"},
		InputSelector:   domain.JSONMap{"resource_ids": []any{"file_agent_a", "file_agent_b"}, "label": "NPH"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"requested_from": "unit_test"},
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	if job.JobID != "data_agent_job_nph_caption" || job.Status != "queued" || job.ResourceCount != 2 || job.ProgressTotal != 2 {
		t.Fatalf("job = %+v, want queued two-resource data-agent job", job)
	}
	if got := job.InputSelector["label"]; got != "NPH" {
		t.Fatalf("job input selector label = %#v, want NPH", got)
	}

	createdEvents, err := store.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 10)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents after create: %v", err)
	}
	if len(createdEvents) != 1 || createdEvents[0].EventType != "data_agent.job.created" || createdEvents[0].Sequence != 1 {
		t.Fatalf("created job events = %+v, want one created event at sequence 1", createdEvents)
	}

	progressEvent, err := store.AppendDataAgentJobEvent(ctx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		EventType:   "data_agent.job.progressed",
		Message:     "Captioned file_agent_a",
		TS:          now.Add(3 * time.Second),
		Metadata:    domain.JSONMap{"completed": float64(1)},
	})
	if err != nil {
		t.Fatalf("AppendDataAgentJobEvent: %v", err)
	}
	if progressEvent.Sequence != 2 {
		t.Fatalf("progress event sequence = %d, want 2", progressEvent.Sequence)
	}

	running, runningEvent, err := store.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID:             job.JobID,
		OwnerUserID:       "alice",
		OwnerOrgID:        "org-a",
		Status:            "running",
		ProgressCompleted: 1,
		ProgressTotal:     2,
		ActorUserID:       "alice",
		ActorOrgID:        "org-a",
		Message:           "Captioned first resource",
		EventMetadata:     domain.JSONMap{"resource_id": "file_agent_a"},
		UpdatedAt:         now.Add(4 * time.Second),
	})
	if err != nil {
		t.Fatalf("UpdateDataAgentJob running: %v", err)
	}
	if running.Status != "running" || running.ProgressCompleted != 1 || running.StartedAt.IsZero() {
		t.Fatalf("running job = %+v, want running progress with start time", running)
	}
	if runningEvent.Sequence != 3 || runningEvent.EventType != "data_agent.job.progressed" {
		t.Fatalf("running event = %+v, want progressed event at sequence 3", runningEvent)
	}

	canceled, canceledEvent, err := store.ControlDataAgentJob(ctx, domain.ControlDataAgentJobInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Action:      "cancel",
		Reason:      "User paused this batch before field upload resumed.",
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		TS:          now.Add(5 * time.Second),
	})
	if err != nil {
		t.Fatalf("ControlDataAgentJob cancel: %v", err)
	}
	if canceled.Status != "canceled" || canceled.Error == "" || canceled.CompletedAt.IsZero() {
		t.Fatalf("canceled job = %+v, want canceled terminal job with reason", canceled)
	}
	if canceledEvent.Sequence != 4 || canceledEvent.EventType != "data_agent.job.canceled" {
		t.Fatalf("canceled event = %+v, want canceled event at sequence 4", canceledEvent)
	}

	retried, retriedEvent, err := store.ControlDataAgentJob(ctx, domain.ControlDataAgentJobInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Action:      "retry",
		Reason:      "Network recovered.",
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		TS:          now.Add(6 * time.Second),
	})
	if err != nil {
		t.Fatalf("ControlDataAgentJob retry: %v", err)
	}
	if retried.Status != "queued" || retried.ProgressCompleted != 0 || retried.Error != "" || !retried.CompletedAt.IsZero() {
		t.Fatalf("retried job = %+v, want reset queued job", retried)
	}
	if retriedEvent.Sequence != 5 || retriedEvent.EventType != "data_agent.job.retried" {
		t.Fatalf("retried event = %+v, want retried event at sequence 5", retriedEvent)
	}

	loaded, err := store.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.JobID != job.JobID || loaded.ProjectID != "nph-study" {
		t.Fatalf("loaded job = %+v, want created job", loaded)
	}

	page, err := store.ListDataAgentJobsForUser(ctx, domain.DataAgentJobListInput{
		UserID:  "alice",
		OrgID:   "org-a",
		Status:  "queued",
		JobType: "caption_resources",
		Limit:   10,
	})
	if err != nil {
		t.Fatalf("ListDataAgentJobsForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Jobs) != 1 || page.Jobs[0].JobID != job.JobID {
		t.Fatalf("job page = %+v, want created queued caption job", page)
	}

	lifecycleEvents, err := store.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 10)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents after lifecycle: %v", err)
	}
	if len(lifecycleEvents) != 5 || lifecycleEvents[4].EventType != "data_agent.job.retried" {
		t.Fatalf("lifecycle events = %+v, want five ordered lifecycle events", lifecycleEvents)
	}

	if _, err := store.GetDataAgentJobForUser(ctx, job.JobID, "bob", "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign GetDataAgentJobForUser error = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreDataAgentJobLeaseLifecycle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 15, 30, 0, 0, time.UTC)

	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_agent_lease_a",
		OriginalName: "lease-a.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    12,
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	job, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_lease",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_agent_lease_a"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}

	lease, leasedJob, event, err := store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-a",
		TTL:         time.Minute,
		Now:         now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("AcquireDataAgentJobLease: %v", err)
	}
	if lease.JobID != job.JobID || lease.WorkerID != "data-agent-worker-a" || lease.LeaseToken == "" {
		t.Fatalf("lease = %+v, want worker-a lease token", lease)
	}
	if !lease.LeaseExpiresAt.Equal(now.Add(62 * time.Second)) {
		t.Fatalf("lease expiry = %s, want now+ttl", lease.LeaseExpiresAt)
	}
	if leasedJob.Status != "running" || leasedJob.StartedAt.IsZero() {
		t.Fatalf("leased job = %+v, want running with started_at", leasedJob)
	}
	if event.Sequence != 2 || event.EventType != "data_agent.job.leased" || event.Metadata["worker_id"] != "data-agent-worker-a" {
		t.Fatalf("lease event = %+v, want leased event at sequence 2", event)
	}

	if _, _, _, err := store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-b",
		TTL:         time.Minute,
		Now:         now.Add(3 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("competing AcquireDataAgentJobLease err = %v, want ErrConflict", err)
	}

	if _, err := store.RenewDataAgentJobLease(ctx, domain.RenewDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		LeaseToken:  "wrong-token",
		TTL:         2 * time.Minute,
		Now:         now.Add(4 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewDataAgentJobLease wrong token err = %v, want ErrConflict", err)
	}
	renewed, err := store.RenewDataAgentJobLease(ctx, domain.RenewDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		LeaseToken:  lease.LeaseToken,
		TTL:         2 * time.Minute,
		Now:         now.Add(4 * time.Second),
	})
	if err != nil {
		t.Fatalf("RenewDataAgentJobLease: %v", err)
	}
	if !renewed.LeaseExpiresAt.Equal(now.Add(124 * time.Second)) {
		t.Fatalf("renewed lease expiry = %s, want renewed now+ttl", renewed.LeaseExpiresAt)
	}

	if err := store.ReleaseDataAgentJobLease(ctx, domain.ReleaseDataAgentJobLeaseInput{JobID: job.JobID, OwnerUserID: "alice", OwnerOrgID: "org-a", LeaseToken: "wrong-token"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("ReleaseDataAgentJobLease wrong token err = %v, want ErrConflict", err)
	}
	if err := store.ReleaseDataAgentJobLease(ctx, domain.ReleaseDataAgentJobLeaseInput{JobID: job.JobID, OwnerUserID: "alice", OwnerOrgID: "org-a", LeaseToken: lease.LeaseToken}); err != nil {
		t.Fatalf("ReleaseDataAgentJobLease: %v", err)
	}
	replacement, _, replacementEvent, err := store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-b",
		TTL:         time.Minute,
		Now:         now.Add(5 * time.Second),
	})
	if err != nil {
		t.Fatalf("AcquireDataAgentJobLease replacement: %v", err)
	}
	if replacement.WorkerID != "data-agent-worker-b" || replacement.LeaseToken == lease.LeaseToken {
		t.Fatalf("replacement lease = %+v, want fresh worker-b lease", replacement)
	}
	if replacementEvent.Sequence != 3 {
		t.Fatalf("replacement lease event = %+v, want sequence 3", replacementEvent)
	}
}

func TestMemoryStoreRecoversExpiredDataAgentJobLeases(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	now := time.Date(2026, 6, 8, 18, 0, 0, 0, time.UTC)

	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_agent_recover_expired",
			OriginalName: "expired.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_agent_recover_active",
			OriginalName: "active.nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	expiredJob, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_recover_expired",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_agent_recover_expired"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob expired: %v", err)
	}
	activeJob, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_recover_active",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_agent_recover_active"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob active: %v", err)
	}
	expiredLease, _, _, err := store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       expiredJob.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-expired",
		TTL:         time.Minute,
		Now:         now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("Acquire expired lease: %v", err)
	}
	activeLease, _, _, err := store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       activeJob.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-active",
		TTL:         10 * time.Minute,
		Now:         now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("Acquire active lease: %v", err)
	}

	result, err := store.RecoverExpiredDataAgentJobLeases(ctx, domain.RecoverExpiredDataAgentJobLeasesInput{
		Now:    now.Add(5 * time.Minute),
		Reason: "automatic expired data-agent lease recovery",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredDataAgentJobLeases: %v", err)
	}
	if result.Checked != 2 || len(result.RequeuedJobs) != 1 || result.RequeuedJobs[0].JobID != expiredJob.JobID {
		t.Fatalf("recovery result = %+v, want one expired job requeued after checking two jobs", result)
	}
	recovered, err := store.GetDataAgentJobForUser(ctx, expiredJob.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser recovered: %v", err)
	}
	if recovered.Status != "queued" || recovered.Error != "" || recovered.ProgressCompleted != 0 || !recovered.CompletedAt.IsZero() {
		t.Fatalf("recovered job = %+v, want reset queued job", recovered)
	}
	if _, err := store.RenewDataAgentJobLease(ctx, domain.RenewDataAgentJobLeaseInput{
		JobID:       expiredJob.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		LeaseToken:  expiredLease.LeaseToken,
		TTL:         time.Minute,
		Now:         now.Add(5*time.Minute + time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("Renew expired recovered lease err = %v, want ErrConflict after lease clear", err)
	}
	if _, err := store.RenewDataAgentJobLease(ctx, domain.RenewDataAgentJobLeaseInput{
		JobID:       activeJob.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		LeaseToken:  activeLease.LeaseToken,
		TTL:         time.Minute,
		Now:         now.Add(5 * time.Minute),
	}); err != nil {
		t.Fatalf("Renew active lease after recovery: %v", err)
	}
	events, err := store.ListDataAgentJobEvents(ctx, expiredJob.JobID, "alice", "org-a", 10)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	last := events[len(events)-1]
	if last.EventType != "data_agent.job.requeued" || last.Metadata["recovery"] != "expired_data_agent_job_lease" || last.Metadata["lease_worker_id"] != "data-agent-worker-expired" {
		t.Fatalf("last recovery event = %+v, want expired lease requeue audit event", last)
	}
}

func TestMemoryStoreCreateAndListUserAccounts(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	user, err := store.CreateUser(ctx, domain.CreateUserInput{
		Email:       "ada@example.org",
		DisplayName: "Ada Lovelace",
		Role:        "admin",
		OrgID:       "local-org",
		Metadata:    domain.JSONMap{"source": "admin_console"},
	})
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if user.UserID == "" {
		t.Fatalf("created user must have user_id")
	}
	if user.Status != "active" {
		t.Fatalf("status = %q, want active", user.Status)
	}

	users, err := store.ListUsers(ctx, 10, "")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 {
		t.Fatalf("users = %d, want 1", len(users))
	}
	got := users[0]
	if got.Email != "ada@example.org" || got.DisplayName != "Ada Lovelace" || got.Role != "admin" || got.OrgID != "local-org" {
		t.Fatalf("unexpected user: %+v", got)
	}
	if got.Metadata["source"] != "admin_console" {
		t.Fatalf("metadata = %#v, want source", got.Metadata)
	}

	filtered, err := store.ListUsers(ctx, 10, "lovelace")
	if err != nil {
		t.Fatalf("ListUsers filtered: %v", err)
	}
	if len(filtered) != 1 || filtered[0].UserID != user.UserID {
		t.Fatalf("filtered users = %+v, want created user", filtered)
	}
}

func TestMemoryStoreDeactivateUserAccount(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	user, err := store.CreateUser(ctx, domain.CreateUserInput{
		Email:       "delete-me@example.org",
		DisplayName: "Delete Me",
		Status:      "active",
	})
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}

	deactivated, err := store.UpdateUserStatus(ctx, user.UserID, "disabled")
	if err != nil {
		t.Fatalf("UpdateUserStatus: %v", err)
	}
	if deactivated.Status != "disabled" {
		t.Fatalf("status = %q, want disabled", deactivated.Status)
	}
	if !deactivated.UpdatedAt.After(user.UpdatedAt) {
		t.Fatalf("updated_at = %s, want after original %s", deactivated.UpdatedAt, user.UpdatedAt)
	}

	users, err := store.ListUsers(ctx, 10, "delete-me")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 || users[0].Status != "disabled" {
		t.Fatalf("users = %+v, want disabled user still visible for audit", users)
	}
}

func TestMemoryStoreCreateAndListOrganizations(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	org, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{
		OrgID:    "allen-institute",
		Name:     "Allen Institute",
		Status:   "active",
		Metadata: domain.JSONMap{"source": "admin_console"},
	})
	if err != nil {
		t.Fatalf("CreateOrganization: %v", err)
	}
	if org.OrgID != "allen-institute" || org.Name != "Allen Institute" || org.Status != "active" {
		t.Fatalf("organization = %+v, want created organization fields", org)
	}

	orgs, err := store.ListOrganizations(ctx, 10, "allen")
	if err != nil {
		t.Fatalf("ListOrganizations: %v", err)
	}
	if len(orgs) != 1 || orgs[0].OrgID != org.OrgID {
		t.Fatalf("organizations = %+v, want created organization", orgs)
	}
	if orgs[0].Metadata["source"] != "admin_console" {
		t.Fatalf("metadata = %#v, want source", orgs[0].Metadata)
	}
	fetched, found, err := store.GetOrganization(ctx, " allen-institute ")
	if err != nil {
		t.Fatalf("GetOrganization: %v", err)
	}
	if !found || fetched.OrgID != org.OrgID || fetched.Metadata["source"] != "admin_console" {
		t.Fatalf("GetOrganization = %+v found=%t, want created organization", fetched, found)
	}
	if _, found, err := store.GetOrganization(ctx, "missing-org"); err != nil || found {
		t.Fatalf("GetOrganization missing found=%t err=%v, want not found without error", found, err)
	}
}

func TestMemoryStoreUpsertsAndListsWorkerHeartbeats(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	started := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	firstBeat := started.Add(10 * time.Second)
	secondBeat := started.Add(70 * time.Second)

	first, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "deepagents-worker-a",
		WorkerKind:      "deepagents",
		Status:          "idle",
		Hostname:        "host-a",
		Version:         "test-version",
		StartedAt:       started,
		LastHeartbeatAt: firstBeat,
		Metadata:        domain.JSONMap{"durable": "ultra-deepagents-worker"},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat first: %v", err)
	}
	if first.WorkerID != "deepagents-worker-a" || first.WorkerKind != "deepagents" || first.Status != "idle" {
		t.Fatalf("first worker = %+v, want deepagents idle record", first)
	}
	if !first.StartedAt.Equal(started) || !first.LastHeartbeatAt.Equal(firstBeat) {
		t.Fatalf("first heartbeat timestamps = %+v", first)
	}

	second, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "deepagents-worker-a",
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    "run_123",
		Hostname:        "host-a",
		Version:         "test-version-2",
		LastHeartbeatAt: secondBeat,
		Metadata:        domain.JSONMap{"active_tasks": 1},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat second: %v", err)
	}
	if second.Status != "busy" || second.CurrentRunID != "run_123" || second.Version != "test-version-2" {
		t.Fatalf("second worker = %+v, want updated busy record", second)
	}
	if !second.StartedAt.Equal(started) {
		t.Fatalf("second started_at = %s, want original %s", second.StartedAt, started)
	}
	if !second.LastHeartbeatAt.Equal(secondBeat) {
		t.Fatalf("second last heartbeat = %s, want %s", second.LastHeartbeatAt, secondBeat)
	}

	fetched, found, err := store.GetWorkerHeartbeat(ctx, "deepagents-worker-a")
	if err != nil {
		t.Fatalf("GetWorkerHeartbeat: %v", err)
	}
	if !found || fetched.WorkerID != "deepagents-worker-a" || fetched.CurrentRunID != "run_123" || fetched.Metadata["active_tasks"] != 1 {
		t.Fatalf("GetWorkerHeartbeat = %+v found=%t, want updated worker", fetched, found)
	}
	if _, found, err := store.GetWorkerHeartbeat(ctx, "missing-worker"); err != nil || found {
		t.Fatalf("GetWorkerHeartbeat missing found=%t err=%v, want not found without error", found, err)
	}

	workers, err := store.ListWorkerHeartbeats(ctx, 10)
	if err != nil {
		t.Fatalf("ListWorkerHeartbeats: %v", err)
	}
	if len(workers) != 1 || workers[0].Status != "busy" || workers[0].Metadata["active_tasks"] != 1 {
		t.Fatalf("workers = %+v, want one updated worker heartbeat", workers)
	}
}

func TestMemoryStoreRejectsDuplicateOrganizationID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	if _, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{OrgID: "smithsonian", Name: "Smithsonian"}); err != nil {
		t.Fatalf("CreateOrganization first: %v", err)
	}
	if _, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{OrgID: "smithsonian", Name: "Smithsonian duplicate"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateOrganization duplicate err = %v, want ErrConflict", err)
	}
}

func TestMemoryStoreRejectsDuplicateUserEmailCaseInsensitive(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	if _, err := store.CreateUser(ctx, domain.CreateUserInput{Email: "Ada@example.org"}); err != nil {
		t.Fatalf("CreateUser first: %v", err)
	}
	if _, err := store.CreateUser(ctx, domain.CreateUserInput{Email: "ada@example.org"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateUser duplicate err = %v, want ErrConflict", err)
	}
}

func TestMemoryStoreListRunEventsAfterSequenceReturnsAscendingPage(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Long trace",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run long autonomous work.",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 5; idx++ {
		if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	events, err := store.ListRunEventsAfter(ctx, run.RunID, 2, 2)
	if err != nil {
		t.Fatalf("ListRunEventsAfter: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	got := []int64{events[0].Sequence, events[1].Sequence}
	want := []int64{3, 4}
	if got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("sequences = %v, want %v", got, want)
	}
}

func TestMemoryStoreRunTokenUsageLedgerIsIdempotentAndFinalizedOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	day := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)

	first, inserted, err := store.RecordRunTokenUsage(ctx, domain.RecordRunTokenUsageInput{
		RunID:        "run-token-ledger",
		UsageEventID: "model-call-1",
		UserID:       "user-token-ledger",
		Model:        "deepseek_v4",
		Day:          day,
		InputTokens:  100,
		OutputTokens: 20,
		TotalTokens:  120,
		OccurredAt:   day,
	})
	if err != nil {
		t.Fatalf("RecordRunTokenUsage first: %v", err)
	}
	if !inserted || first.TotalTokens != 120 {
		t.Fatalf("first usage = %+v inserted=%v, want inserted total 120", first, inserted)
	}
	duplicate, inserted, err := store.RecordRunTokenUsage(ctx, domain.RecordRunTokenUsageInput{
		RunID:        "run-token-ledger",
		UsageEventID: "model-call-1",
		UserID:       "user-token-ledger",
		Model:        "deepseek_v4",
		Day:          day,
		InputTokens:  100,
		OutputTokens: 20,
		TotalTokens:  120,
		OccurredAt:   day,
	})
	if err != nil {
		t.Fatalf("RecordRunTokenUsage duplicate: %v", err)
	}
	if inserted || duplicate.TotalTokens != 120 {
		t.Fatalf("duplicate usage = %+v inserted=%v, want existing total 120", duplicate, inserted)
	}
	if _, inserted, err := store.FinalizeRunTokenUsage(ctx, domain.FinalizeRunTokenUsageInput{
		RunID:       "run-token-ledger",
		CompletedAt: day,
	}); err != nil {
		t.Fatalf("FinalizeRunTokenUsage first: %v", err)
	} else if !inserted {
		t.Fatalf("first finalize inserted=%v, want true", inserted)
	}
	if _, inserted, err := store.FinalizeRunTokenUsage(ctx, domain.FinalizeRunTokenUsageInput{
		RunID:       "run-token-ledger",
		CompletedAt: day,
	}); err != nil {
		t.Fatalf("FinalizeRunTokenUsage duplicate: %v", err)
	} else if inserted {
		t.Fatalf("duplicate finalize inserted=%v, want false", inserted)
	}

	stats, err := store.GetUserTokenUsageStats(ctx, "user-token-ledger")
	if err != nil {
		t.Fatalf("GetUserTokenUsageStats: %v", err)
	}
	if stats.TotalTokens != 120 {
		t.Fatalf("stats = %+v, want total 120", stats)
	}
	daily, err := store.ListUserTokenUsageDaily(ctx, "user-token-ledger", time.Time{})
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily: %v", err)
	}
	if len(daily) != 1 || daily[0].TotalTokens != 120 || daily[0].RunCount != 1 {
		t.Fatalf("daily = %+v, want one row total 120 run_count 1", daily)
	}
}

func TestMemoryStoreTokenUsageUsesOccurredAtWhenDayIsOmitted(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	occurredAt := time.Date(2031, 4, 5, 18, 30, 0, 0, time.FixedZone("PDT", -7*60*60))
	wantDay := time.Date(2031, 4, 6, 0, 0, 0, 0, time.UTC)

	if err := store.RecordUserTokenUsage(ctx, domain.RecordUserTokenUsageInput{
		UserID:       "user-token-day",
		InputTokens:  7,
		OutputTokens: 3,
		TotalTokens:  10,
		OccurredAt:   occurredAt,
	}); err != nil {
		t.Fatalf("RecordUserTokenUsage: %v", err)
	}
	if _, inserted, err := store.RecordRunTokenUsage(ctx, domain.RecordRunTokenUsageInput{
		RunID:        "run-token-day",
		UsageEventID: "usage-token-day",
		UserID:       "user-token-day",
		InputTokens:  11,
		OutputTokens: 4,
		TotalTokens:  15,
		OccurredAt:   occurredAt,
	}); err != nil {
		t.Fatalf("RecordRunTokenUsage: %v", err)
	} else if !inserted {
		t.Fatalf("RecordRunTokenUsage inserted=%v, want true", inserted)
	}

	daily, err := store.ListUserTokenUsageDaily(ctx, "user-token-day", time.Time{})
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily: %v", err)
	}
	if len(daily) != 1 {
		t.Fatalf("daily rows = %d, want 1: %+v", len(daily), daily)
	}
	if !daily[0].Day.Equal(wantDay) {
		t.Fatalf("daily day = %s, want %s", daily[0].Day.Format(time.RFC3339), wantDay.Format(time.RFC3339))
	}
	if daily[0].TotalTokens != 25 {
		t.Fatalf("daily total = %d, want 25", daily[0].TotalTokens)
	}
}

func TestMemoryStoreUpdateRunStatusKeepsTerminalRunImmutable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Terminal run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run a long analysis.",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	completed, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "final answer", "")
	if err != nil {
		t.Fatalf("UpdateRunStatus succeeded: %v", err)
	}
	if completed.CompletedAt == nil {
		t.Fatalf("completed run must have completed_at set")
	}

	reopened, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", "")
	if err != nil {
		t.Fatalf("UpdateRunStatus stale running: %v", err)
	}
	if reopened.Status != domain.RunStatusSucceeded {
		t.Fatalf("status = %s, want terminal succeeded to be preserved", reopened.Status)
	}
	if reopened.ResponseText != "final answer" {
		t.Fatalf("response text = %q, want first terminal response preserved", reopened.ResponseText)
	}
	if reopened.CompletedAt == nil || !reopened.CompletedAt.Equal(*completed.CompletedAt) {
		t.Fatalf("completed_at changed after stale update: before=%v after=%v", completed.CompletedAt, reopened.CompletedAt)
	}

	failed, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusFailed, "", "late failure")
	if err != nil {
		t.Fatalf("UpdateRunStatus stale failure: %v", err)
	}
	if failed.Status != domain.RunStatusSucceeded || failed.Error != "" {
		t.Fatalf("stale failure mutated terminal run: %+v", failed)
	}
}

func TestMemoryStoreCompleteRunRepairsSucceededRunMissingResponseText(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Terminal repair"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Repair missing terminal response.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Repair missing terminal response."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus succeeded with empty response: %v", err)
	}

	repaired, err := store.CompleteRun(ctx, domain.CompleteRunInput{
		RunID:        run.RunID,
		ResponseText: "Recovered final answer.",
	})
	if err != nil {
		t.Fatalf("CompleteRun repair: %v", err)
	}
	if repaired.Status != domain.RunStatusSucceeded || repaired.ResponseText != "Recovered final answer." {
		t.Fatalf("repaired run = %+v, want succeeded with recovered response text", repaired)
	}
	messages, err := store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("messages = %d, want %d user+assistant messages: %+v", got, want, messages)
	}
	if messages[1].Role != "assistant" || messages[1].Content != "Recovered final answer." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want recovered response owned by run", messages[1])
	}
}

func TestMemoryStoreRunLeasePreventsConcurrentWorkersAndCanExpire(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Lease thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long autonomous run",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	first, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Minute,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease first: %v", err)
	}
	if first.LeaseToken == "" || first.WorkerID != "worker-a" || !first.LeaseExpiresAt.Equal(now.Add(time.Minute)) {
		t.Fatalf("first lease = %+v, want worker-a token expiring after ttl", first)
	}
	updatedRun, err := store.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updatedRun.Status != domain.RunStatusRunning || updatedRun.StartedAt == nil {
		t.Fatalf("claimed run = %+v, want running with started_at", updatedRun)
	}

	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Minute,
		Now:      now.Add(30 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("AcquireRunLease competing err = %v, want ErrConflict", err)
	}

	second, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      2 * time.Minute,
		Now:      now.Add(2 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease after expiry: %v", err)
	}
	if second.WorkerID != "worker-b" || second.LeaseToken == first.LeaseToken || !second.LeaseExpiresAt.Equal(now.Add(4*time.Minute)) {
		t.Fatalf("second lease = %+v, want replacement worker-b lease", second)
	}
}

func TestMemoryStoreRunLeaseRenewAndReleaseRequireToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Lease token"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "user-1", Goal: "work"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	lease, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Minute,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	if _, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: "wrong-token",
		TTL:        time.Minute,
		Now:        now.Add(30 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewRunLease wrong token err = %v, want ErrConflict", err)
	}
	renewed, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        5 * time.Minute,
		Now:        now.Add(30 * time.Second),
	})
	if err != nil {
		t.Fatalf("RenewRunLease: %v", err)
	}
	if !renewed.LeaseExpiresAt.Equal(now.Add(30*time.Second + 5*time.Minute)) {
		t.Fatalf("renewed lease expiry = %s, want now+ttl", renewed.LeaseExpiresAt)
	}

	if err := store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{RunID: run.RunID, LeaseToken: "wrong-token"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("ReleaseRunLease wrong token err = %v, want ErrConflict", err)
	}
	if err := store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{RunID: run.RunID, LeaseToken: lease.LeaseToken}); err != nil {
		t.Fatalf("ReleaseRunLease: %v", err)
	}
	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Minute,
		Now:      now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease after release: %v", err)
	}
}

func TestMemoryStoreClearRunLeaseEvictsAnyActiveToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Clear lease"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "user-1", Goal: "recover"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	lease, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Hour,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	cleared, ok, err := store.ClearRunLease(ctx, run.RunID)
	if err != nil {
		t.Fatalf("ClearRunLease: %v", err)
	}
	if !ok || cleared.LeaseToken != lease.LeaseToken || cleared.WorkerID != "worker-a" {
		t.Fatalf("cleared lease = %+v ok=%v, want worker-a lease", cleared, ok)
	}
	if _, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
		Now:        now.Add(time.Minute),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewRunLease after clear err = %v, want ErrConflict", err)
	}
	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Hour,
		Now:      now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease replacement: %v", err)
	}
}
