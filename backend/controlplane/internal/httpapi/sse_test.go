package httpapi

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestRunEventsStreamEmitsRunEventEnvelope(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "stream"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "stream",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stream"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	if !strings.Contains(body, "event: run_event") {
		t.Fatalf("stream body missing run_event: %s", body)
	}
	if !strings.Contains(body, "run.accepted") {
		t.Fatalf("stream body missing run.accepted: %s", body)
	}

	scanner := bufio.NewScanner(strings.NewReader(body))
	foundData := false
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "data:") {
			foundData = true
		}
	}
	if !foundData {
		t.Fatalf("stream body missing data lines: %s", body)
	}
}

func TestRunEventsStreamDoesNotMissEventBetweenReplayAndSubscription(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "stream race"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "stream race",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stream race"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	var fired atomic.Bool
	storeWithRace := &listRunEventsHookStore{
		Store: mem,
		afterList: func() {
			if !fired.CompareAndSwap(false, true) {
				return
			}
			event, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt-racy-delta",
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "message.delta",
				Message:   "racy chunk",
			})
			if err != nil {
				t.Errorf("AppendRunEvent in race hook: %v", err)
				return
			}
			if err := bus.PublishRunEvent(ctx, event); err != nil {
				t.Errorf("PublishRunEvent in race hook: %v", err)
			}
		},
	}
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: storeWithRace, Bus: bus})

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	if !strings.Contains(body, "racy chunk") {
		t.Fatalf("stream missed event persisted between replay snapshot and live subscription: %s", body)
	}
}

func TestRunEventsStreamCatchesUpPersistedEventsWithoutLocalFanout(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "cross instance stream"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "cross instance stream",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "cross instance stream"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true&after_sequence=0", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(50 * time.Millisecond)
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-cross-instance-delta",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		Message:   "persisted by another control plane",
	}); err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}

	time.Sleep(1200 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	if !strings.Contains(body, "persisted by another control plane") {
		t.Fatalf("stream did not catch up event persisted without local fanout: %s", body)
	}
}

func TestRunEventsStreamDeliversSequenceOrderWhenBusCarriesPartialEvents(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "ordered stream"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "ordered stream",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "ordered stream"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true&after_sequence=0", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()
	time.Sleep(40 * time.Millisecond)

	// Simulate a second control-plane replica ingesting deltas 2..5: they are
	// persisted to the shared store but never reach this replica's local bus.
	// Only the final delta is ingested locally and fanned out on the bus.
	var lastEvent domain.RunEventRecord
	for index := 1; index <= 5; index++ {
		event, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "evt-ordered-" + strconv.Itoa(index),
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk " + strconv.Itoa(index),
		})
		if err != nil {
			t.Fatalf("AppendRunEvent %d: %v", index, err)
		}
		lastEvent = event
	}
	if err := bus.PublishRunEvent(ctx, lastEvent); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}

	// Well under the 1s periodic catch-up: ordering must not depend on it.
	time.Sleep(250 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	var sequences []int64
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if !strings.Contains(payload, "\"sequence\"") {
			continue
		}
		marker := "\"sequence\":"
		start := strings.Index(payload, marker)
		if start == -1 {
			continue
		}
		rest := payload[start+len(marker):]
		end := strings.IndexAny(rest, ",}")
		if end == -1 {
			continue
		}
		sequence, err := strconv.ParseInt(strings.TrimSpace(rest[:end]), 10, 64)
		if err != nil {
			continue
		}
		sequences = append(sequences, sequence)
	}
	if len(sequences) != 6 {
		t.Fatalf("delivered sequences = %v, want all 6 events (1 accepted + 5 deltas); body=%s", sequences, body)
	}
	for index, sequence := range sequences {
		if int64(index+1) != sequence {
			t.Fatalf("sequence order = %v, want strictly increasing 1..6 with no gaps", sequences)
		}
	}
}

type listRunEventsHookStore struct {
	runcontrol.Store
	afterList   func()
	mu          sync.Mutex
	afterLimits []int
}

func (s *listRunEventsHookStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	events, err := s.Store.ListRunEvents(ctx, runID, limit)
	if err == nil && s.afterList != nil {
		s.afterList()
	}
	return events, err
}

func (s *listRunEventsHookStore) ListRunEventsAfter(ctx context.Context, runID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	s.mu.Lock()
	s.afterLimits = append(s.afterLimits, limit)
	s.mu.Unlock()
	events, err := s.Store.ListRunEventsAfter(ctx, runID, afterSequence, limit)
	if err == nil && s.afterList != nil {
		s.afterList()
	}
	return events, err
}

func (s *listRunEventsHookStore) recordedAfterLimits() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	limits := make([]int, len(s.afterLimits))
	copy(limits, s.afterLimits)
	return limits
}

func TestRunEventsStreamRespectsAfterSequenceReplayCursor(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "stream cursor"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "stream cursor",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stream cursor"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 4; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true&limit=2&after_sequence=3", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	if strings.Contains(body, `"sequence":1`) || strings.Contains(body, `"sequence":2`) || strings.Contains(body, `"sequence":3`) {
		t.Fatalf("stream replay included events before cursor: %s", body)
	}
	if !strings.Contains(body, `"sequence":4`) || !strings.Contains(body, `"sequence":5`) {
		t.Fatalf("stream replay missing cursor events: %s", body)
	}
}

func TestRunEventsStreamPagesFullCursorReplayBeforeLiveStreaming(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "long replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "long replay",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long replay"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 5; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true&limit=2&after_sequence=0", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	for sequence := 1; sequence <= 6; sequence++ {
		if !strings.Contains(body, `"sequence":`+strconv.Itoa(sequence)) {
			t.Fatalf("stream replay missing sequence %d from full cursor replay: %s", sequence, body)
		}
	}
}

func TestRunEventsStreamCapsReplayPageSizeWhileRecoveringFullCursor(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	storeWithLimits := &listRunEventsHookStore{Store: mem}
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: storeWithLimits, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "bounded replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "bounded replay",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bounded replay"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 3; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true&limit=1000000&after_sequence=0", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	limits := storeWithLimits.recordedAfterLimits()
	if len(limits) == 0 {
		t.Fatalf("expected cursor replay to query persisted events")
	}
	for _, limit := range limits {
		if limit > 1000 {
			t.Fatalf("cursor replay used unbounded page size %d; limits=%v", limit, limits)
		}
	}

	body := rec.Body.String()
	for sequence := 1; sequence <= 4; sequence++ {
		if !strings.Contains(body, `"sequence":`+strconv.Itoa(sequence)) {
			t.Fatalf("bounded full replay missing sequence %d: %s", sequence, body)
		}
	}
}
