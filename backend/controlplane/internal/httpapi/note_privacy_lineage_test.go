package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestCreateRunHTTPKeepsNotePrivacyLineageServerAuthored(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	privateThread, err := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "private"})
	if err != nil {
		t.Fatalf("CreateThread private: %v", err)
	}
	if _, err := runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: privateThread.ThreadID,
		UserID:   "ada",
		Goal:     "Use my Note",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode: domain.NoteAccessModeSearch,
		}),
	}); err != nil {
		t.Fatalf("CreateRun Note: %v", err)
	}

	ordinaryThread, err := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "ordinary"})
	if err != nil {
		t.Fatalf("CreateThread ordinary: %v", err)
	}
	router := NewRouter(ServerDeps{Runs: runs, Store: mem})

	spoofBody := fmt.Sprintf(`{"goal":"ordinary","metadata":{"%s":true}}`, domain.NotePrivacyLineageMetadataKey)
	spoofRec := notesRequest(router, http.MethodPost, "/v2/threads/"+ordinaryThread.ThreadID+"/runs", "ada", spoofBody)
	if spoofRec.Code != http.StatusOK {
		t.Fatalf("ordinary spoof create = %d body=%s", spoofRec.Code, spoofRec.Body.String())
	}
	var ordinaryRun domain.RunRecord
	if err := json.Unmarshal(spoofRec.Body.Bytes(), &ordinaryRun); err != nil {
		t.Fatalf("decode ordinary run: %v", err)
	}
	if domain.RunHasNotePrivacyLineage(ordinaryRun) {
		t.Fatalf("HTTP metadata spoof minted privacy lineage: %+v", ordinaryRun.Metadata)
	}

	followupBody := fmt.Sprintf(`{"goal":"explain that answer","metadata":{"%s":false}}`, domain.NotePrivacyLineageMetadataKey)
	followupRec := notesRequest(router, http.MethodPost, "/v2/threads/"+privateThread.ThreadID+"/runs", "ada", followupBody)
	if followupRec.Code != http.StatusOK {
		t.Fatalf("private follow-up create = %d body=%s", followupRec.Code, followupRec.Body.String())
	}
	var followupRun domain.RunRecord
	if err := json.Unmarshal(followupRec.Body.Bytes(), &followupRun); err != nil {
		t.Fatalf("decode follow-up run: %v", err)
	}
	if !domain.RunHasNotePrivacyLineage(followupRun) || domain.RunHasNoteAccessSelection(followupRun) {
		t.Fatalf("HTTP follow-up metadata = %+v, want inherited lineage without Note scope", followupRun.Metadata)
	}
}
