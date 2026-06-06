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
