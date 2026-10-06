package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type fakeBisqueModuleState struct {
	mu            sync.Mutex
	executeCalls  int
	executeBody   string
	executeType   string
	executeUser   string
	executePass   string
	executeAuthed bool
}

// newFakeBisqueModuleServer mirrors bisque2's real shapes: module_service lists
// only active registrations while data_service also holds a retired duplicate,
// module inputs carry identity attributes and a <template> element, and module
// runs echo system inputs (including a live bisque_token).
func newFakeBisqueModuleServer(t *testing.T) (*httptest.Server, *fakeBisqueModuleState) {
	t.Helper()
	state := &fakeBisqueModuleState{}
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		root := server.URL
		w.Header().Set("Content-Type", "text/xml")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/module_service/":
			fmt.Fprintf(w, `<resource uri="%[1]s/module_service/">`+
				`<module uri="%[1]s/data_service/00-mod" resource_uniq="00-mod" name="PrairieDogDetection" type="runtime" value="http://engine:27000/engine_service/PrairieDogDetection"/>`+
				`<module uri="%[1]s/data_service/00-edge" resource_uniq="00-edge" name="EdgeDetection" type="runtime"/>`+
				`</resource>`, root)
		case r.Method == http.MethodGet && r.URL.Path == "/data_service/module":
			if r.URL.Query().Get("view") != "deep" {
				t.Errorf("module definitions fetched without view=deep: %s", r.URL.RawQuery)
			}
			fmt.Fprintf(w, `<resource uri="%[1]s/data_service/module">`+
				`<module uri="%[1]s/data_service/00-mod" resource_uniq="00-mod" name="PrairieDogDetection" type="runtime" ts="2026-07-08T20:37:14">`+
				`<tag name="title" value="PrairieDogDetection"/>`+
				`<tag name="description" value="This module detects prairie dogs and their burroughs"/>`+
				`<tag name="authors" value="BisQue Team"/>`+
				`<tag name="inputs" uri="%[1]s/data_service/00-mod/tag/1" owner="%[1]s/data_service/00-owner" permission="published">`+
				`<tag name="Input Image" type="resource" uri="%[1]s/data_service/00-mod/tag/2"><template uri="%[1]s/data_service/00-mod/tag/2/template/3">`+
				`<tag name="label" value="Input Image"/><tag name="accepted_type" value="image"/><tag name="accepted_type" value="dataset"/></template></tag>`+
				`<tag name="Annotation Options" type="combo" value="Create new annotations"/>`+
				`<tag name="Annotation Name" type="annotation_attr"><template><tag name="allowBlank" value="false"/></template></tag>`+
				`<tag name="weights" type="resource"><template><tag name="label" value="Model Weights (optional)"/>`+
				`<tag name="accepted_type" value="file"/><tag name="allow_blank" value="true"/></template></tag>`+
				`<tag name="mex_url" type="system-input_resource"/>`+
				`<tag name="bisque_token" type="system-input_resource"/>`+
				`</tag>`+
				`<tag name="outputs"><tag name="NonImage"><tag name="csv_file" type="file"/></tag></tag>`+
				`</module>`+
				`<module uri="%[1]s/data_service/00-retired" resource_uniq="00-retired" name="AnnotatePrairies" type="runtime">`+
				`<tag name="description" value="Retired registration"/></module>`+
				`<module uri="%[1]s/data_service/00-edge" resource_uniq="00-edge" name="EdgeDetection" type="runtime">`+
				`<tag name="description" value="This module finds edges in images"/>`+
				`<tag name="inputs"><tag name="Input Image" type="resource"/></tag></module>`+
				`</resource>`, root)
		case r.Method == http.MethodGet && r.URL.Path == "/module_service/PrairieDogDetection/help":
			w.Header().Set("Content-Type", "text/html")
			_, _ = io.WriteString(w, `<html><head><style>p{}</style></head><body><h1>About</h1>`+
				`<p>Detects prairie dogs &amp; burrows with YOLO-v5.</p><script>track()</script></body></html>`)
		case r.Method == http.MethodGet && r.URL.Path == "/data_service/mex":
			if got := r.URL.Query().Get("tag_query"); got != "@name:PrairieDogDetection" {
				t.Errorf("example-run search tag_query = %q", got)
			}
			fmt.Fprintf(w, `<resource><mex uri="%[1]s/data_service/00-run" resource_uniq="00-run" name="PrairieDogDetection" value="FINISHED"/></resource>`, root)
		case r.Method == http.MethodGet && r.URL.Path == "/data_service/00-run":
			fmt.Fprintf(w, `<mex uri="%[1]s/data_service/00-run" resource_uniq="00-run" name="PrairieDogDetection" value="FINISHED" type="%[1]s/data_service/00-mod">`+
				`<tag name="inputs"><tag name="Input Image" type="image" value="%[1]s/data_service/00-img"/>`+
				`<tag name="weights" type="file" value="%[1]s/data_service/00-w"/>`+
				`<tag name="mex_url" type="system-input_resource" value="%[1]s/data_service/00-run"/>`+
				`<tag name="bisque_token" type="system-input_resource" value="SECRET-FINISHED-TOKEN"/></tag>`+
				`<tag name="outputs"><tag name="csv_file" type="file" value="%[1]s/data_service/00-csv"/></tag></mex>`, root)
		case r.Method == http.MethodGet && r.URL.Path == "/data_service/00-img":
			fmt.Fprintf(w, `<image uri="%[1]s/data_service/00-img" resource_uniq="00-img" name="demo.jpg"/>`, root)
		case r.Method == http.MethodGet && r.URL.Path == "/data_service/00-w":
			fmt.Fprintf(w, `<file uri="%[1]s/data_service/00-w" resource_uniq="00-w" name="prairie_2024.pt"/>`, root)
		case r.Method == http.MethodPost && r.URL.Path == "/module_service/PrairieDogDetection/execute":
			body, _ := io.ReadAll(r.Body)
			user, pass, ok := r.BasicAuth()
			state.mu.Lock()
			state.executeCalls++
			state.executeBody = string(body)
			state.executeType = r.Header.Get("Content-Type")
			state.executeUser, state.executePass, state.executeAuthed = user, pass, ok
			state.mu.Unlock()
			// module_service remaps the new run's uri to /module_service/mex/<uniq>.
			fmt.Fprintf(w, `<mex uri="%[1]s/module_service/mex/00-new" resource_uniq="00-new" name="PrairieDogDetection" value="DISPATCH" type="%[1]s/data_service/00-mod">`+
				`<tag name="inputs"><tag name="Input Image" type="image" value="%[1]s/data_service/00-img"/>`+
				`<tag name="bisque_token" type="system-input_resource" value="LIVE-RUN-TOKEN"/></tag></mex>`, root)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, state
}

func newBisqueModuleRouter(t *testing.T, server *httptest.Server) http.Handler {
	t.Helper()
	return NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL: server.URL, AllowedRoots: []string{server.URL}, HTTPClient: server.Client(),
			DevUsername: "demo-user", DevPassword: "demo-pass", UploadRoot: t.TempDir(), MaxImportSize: 8 << 20,
		}),
	})
}

func postModuleJSON(router http.Handler, path string, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func TestV2BisqueModulesListsOnlyRegisteredModulesRanked(t *testing.T) {
	t.Parallel()
	server, _ := newFakeBisqueModuleServer(t)
	router := newBisqueModuleRouter(t, server)

	body := postJSON(t, router, "/v2/bisque/modules", `{"query":"detect prairie dogs in aerial images"}`)
	if body["total"].(float64) != 2 {
		t.Fatalf("total = %v, want 2 (retired data_service registrations are excluded)", body["total"])
	}
	modules := body["modules"].([]any)
	first := modules[0].(map[string]any)
	if first["name"] != "PrairieDogDetection" || first["match_score"].(float64) < 2 {
		t.Fatalf("first module = %#v, want PrairieDogDetection ranked first", first)
	}
	if !strings.Contains(first["description"].(string), "prairie dogs") {
		t.Fatalf("description = %#v", first["description"])
	}
	inputs := first["inputs"].([]any)
	if len(inputs) != 4 {
		t.Fatalf("inputs = %#v, want 4 user-facing inputs (system inputs hidden)", inputs)
	}
	image := inputs[0].(map[string]any)
	template := image["template"].(map[string]any)
	if image["name"] != "Input Image" || template["accepted_type"] != "image, dataset" {
		t.Fatalf("Input Image = %#v, want repeated accepted_type values joined", image)
	}
	if outputs := first["outputs"].([]any); len(outputs) != 1 || outputs[0] != "NonImage" {
		t.Fatalf("outputs = %#v", outputs)
	}
}

func TestV2BisqueModuleDescribeAddsHelpAndRedactedExampleRun(t *testing.T) {
	t.Parallel()
	server, _ := newFakeBisqueModuleServer(t)
	router := newBisqueModuleRouter(t, server)

	rec := postModuleJSON(router, "/v2/bisque/module", `{"name":"prairiedogdetection"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "SECRET-FINISHED-TOKEN") {
		t.Fatalf("describe leaked a run's bisque_token: %s", rec.Body.String())
	}
	var out struct {
		Module BisqueModule `json:"module"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Module.Name != "PrairieDogDetection" {
		t.Fatalf("name = %q, want case-insensitive match", out.Module.Name)
	}
	if out.Module.Help != "About Detects prairie dogs & burrows with YOLO-v5." {
		t.Fatalf("help = %q (script/style must be stripped, entities decoded)", out.Module.Help)
	}
	if out.Module.ExampleRun == nil || out.Module.ExampleRun.Status != "FINISHED" {
		t.Fatalf("example_run = %#v", out.Module.ExampleRun)
	}
	var weights string
	for _, input := range out.Module.ExampleRun.Inputs {
		if input.Name == "weights" {
			weights = input.ResourceURI
		}
		if input.Name == "mex_url" || input.Name == "bisque_token" {
			t.Fatalf("example_run exposes system input %q", input.Name)
		}
	}
	if weights != server.URL+"/data_service/00-w" {
		t.Fatalf("example weights = %q", weights)
	}
}

func TestV2BisqueModuleExecuteSubmitsMexByReference(t *testing.T) {
	t.Parallel()
	server, state := newFakeBisqueModuleServer(t)
	router := newBisqueModuleRouter(t, server)

	rec := postModuleJSON(router, "/v2/bisque/module-execute", fmt.Sprintf(
		`{"module_name":"PrairieDogDetection","inputs":{"input image":%q,"weights":"00-w","Annotation Name":"ultra_demo"}}`,
		server.URL+"/data_service/00-img"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "LIVE-RUN-TOKEN") {
		t.Fatalf("execute response leaked the live run token: %s", rec.Body.String())
	}
	var out bisqueModuleExecuteResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Status != "DISPATCH" || out.MexURI != server.URL+"/data_service/00-new" ||
		out.ClientViewURL != server.URL+"/client_service/view?resource="+server.URL+"/data_service/00-new" {
		t.Fatalf("response = %#v, want the canonical data_service run URI", out)
	}

	state.mu.Lock()
	defer state.mu.Unlock()
	if state.executeCalls != 1 || !state.executeAuthed || state.executeUser != "demo-user" || state.executePass != "demo-pass" {
		t.Fatalf("execute calls=%d auth=%v user=%q", state.executeCalls, state.executeAuthed, state.executeUser)
	}
	if !strings.HasPrefix(state.executeType, "text/xml") {
		t.Fatalf("execute content-type = %q", state.executeType)
	}
	for _, leaked := range []string{"uri=", "owner=", "permission=", "template"} {
		if strings.Contains(state.executeBody, leaked) {
			t.Fatalf("MEX body carries definition-only %q: %s", leaked, state.executeBody)
		}
	}
	var mex bisqueXMLNode
	if err := xml.Unmarshal([]byte(state.executeBody), &mex); err != nil || mex.XMLName.Local != "mex" || len(mex.Children) != 1 {
		t.Fatalf("MEX body = %s (err %v)", state.executeBody, err)
	}
	got := map[string][2]string{}
	order := []string{}
	for _, input := range mex.Children[0].Children {
		got[input.attr("name")] = [2]string{input.attr("type"), input.attr("value")}
		order = append(order, input.attr("name"))
	}
	want := map[string][2]string{
		"Input Image":        {"image", server.URL + "/data_service/00-img"},
		"Annotation Options": {"combo", "Create new annotations"},
		"Annotation Name":    {"annotation_attr", "ultra_demo"},
		"weights":            {"file", server.URL + "/data_service/00-w"},
		"mex_url":            {"system-input_resource", ""},
		"bisque_token":       {"system-input_resource", ""},
	}
	if len(got) != len(want) {
		t.Fatalf("MEX inputs = %v, want %v", got, want)
	}
	for name, expected := range want {
		if got[name] != expected {
			t.Fatalf("MEX input %q = %v, want %v (order %v)", name, got[name], expected, order)
		}
	}
}

func TestV2BisqueModuleExecuteRejectsUnsafeOrInvalidInputs(t *testing.T) {
	t.Parallel()
	server, state := newFakeBisqueModuleServer(t)
	router := newBisqueModuleRouter(t, server)
	image := server.URL + "/data_service/00-img"

	cases := map[string]struct {
		body string
		want string
	}{
		"outside root": {
			body: `{"module_name":"PrairieDogDetection","inputs":{"Input Image":"https://elsewhere.example/data_service/00-x","weights":"00-w"}}`,
			want: "configured BisQue server",
		},
		"unknown input": {
			body: fmt.Sprintf(`{"module_name":"PrairieDogDetection","inputs":{"Input Image":%q,"weights":"00-w","threshold":"0.5"}}`, image),
			want: "unknown inputs",
		},
		"system input": {
			body: fmt.Sprintf(`{"module_name":"PrairieDogDetection","inputs":{"Input Image":%q,"weights":"00-w","bisque_token":"forged"}}`, image),
			want: "unknown inputs",
		},
		"missing required": {
			body: `{"module_name":"PrairieDogDetection","inputs":{"weights":"00-w"}}`,
			want: "missing required inputs for PrairieDogDetection: Input Image, Annotation Name",
		},
		"unreadable resource": {
			body: `{"module_name":"PrairieDogDetection","inputs":{"Input Image":"00-missing","weights":"00-w"}}`,
			want: "could not be read on BisQue",
		},
		"unknown module": {
			body: `{"module_name":"NoSuchModule","inputs":{}}`,
			want: "not registered",
		},
	}
	for name, tc := range cases {
		rec := postModuleJSON(router, "/v2/bisque/module-execute", tc.body)
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), tc.want) {
			t.Fatalf("%s: status=%d body=%s, want 400 containing %q", name, rec.Code, rec.Body.String(), tc.want)
		}
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.executeCalls != 0 {
		t.Fatalf("invalid requests reached BisQue execute %d times", state.executeCalls)
	}
}

// The real PrairieDogDetection marks weights allow_blank=true ("Model Weights
// (optional)"); omitting it must submit a typed, valueless input, not fail.
func TestV2BisqueModuleExecuteHonorsOptionalTemplateInputs(t *testing.T) {
	t.Parallel()
	server, state := newFakeBisqueModuleServer(t)
	router := newBisqueModuleRouter(t, server)

	rec := postModuleJSON(router, "/v2/bisque/module-execute", fmt.Sprintf(
		`{"module_name":"PrairieDogDetection","inputs":{"Input Image":%q,"Annotation Name":"ultra_demo"}}`,
		server.URL+"/data_service/00-img"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	var mex bisqueXMLNode
	if err := xml.Unmarshal([]byte(state.executeBody), &mex); err != nil {
		t.Fatalf("MEX body: %v", err)
	}
	for _, input := range mex.Children[0].Children {
		if input.attr("name") == "weights" {
			if input.attr("type") != "resource" || input.attr("value") != "" {
				t.Fatalf("optional weights = type %q value %q, want typed and blank", input.attr("type"), input.attr("value"))
			}
			return
		}
	}
	t.Fatalf("optional weights input missing from MEX: %s", state.executeBody)
}

func TestWorkerBisqueModuleExecuteRequiresRunModuleIntentAndReplays(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	server, state := newFakeBisqueModuleServer(t)
	memory := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{9}, 32), "module-test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	credentials := NewPersistentBisqueCredentialStore(memory, cipher, server.URL)
	sessionID, err := credentials.PutLinked(ctx, BisqueCredentialLinkInput{
		Credentials: BisqueCredentials{Username: "ecologist", Password: "linked-secret"},
		UserID:      "ecologist-1",
		OrgID:       "field-lab",
		RootURL:     server.URL,
	})
	if err != nil {
		t.Fatalf("PutLinked: %v", err)
	}
	router := NewRouter(ServerDeps{
		Version: "test-version", Store: memory, UploadRoot: t.TempDir(), ArtifactRoot: t.TempDir(),
		WorkerToken: "worker-secret", BisqueCredentials: credentials,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL: server.URL, AllowedRoots: []string{server.URL}, HTTPClient: server.Client(),
			UploadRoot: t.TempDir(), MaxImportSize: 8 << 20,
		}),
	})
	thread, err := memory.CreateThread(ctx, domain.CreateThreadInput{UserID: "ecologist-1", Title: "module run"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	newRun := func(intents ...domain.RemoteMutationIntent) (domain.RunRecord, domain.RunLeaseRecord) {
		run, err := memory.CreateRun(ctx, domain.CreateRunInput{
			ThreadID: thread.ThreadID, UserID: "ecologist-1", Goal: "run prairie dog detection",
			Metadata: testRunAuthorityMetadata("ecologist-1", "field-lab", server.URL, sessionID, intents...),
		})
		if err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		return run, startTestWorkerRunLease(t, memory, run.RunID, "worker-a")
	}
	post := func(run domain.RunRecord, lease domain.RunLeaseRecord, path string, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		setTestWorkerRunHeaders(req, run.RunID, "worker-a", lease.LeaseToken, sessionID)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}
	execute := fmt.Sprintf(`{"module_name":"PrairieDogDetection","inputs":{"Input Image":%q,"weights":"00-w","Annotation Name":"ultra_demo"}}`,
		server.URL+"/data_service/00-img")

	readOnlyRun, readOnlyLease := newRun(domain.RemoteMutationIntentBisqueUpload)
	if rec := post(readOnlyRun, readOnlyLease, "/v2/bisque/modules", `{}`); rec.Code != http.StatusOK {
		t.Fatalf("worker list modules status = %d body=%s (listing is read-only)", rec.Code, rec.Body.String())
	}
	if rec := post(readOnlyRun, readOnlyLease, "/v2/bisque/module-execute", execute); rec.Code != http.StatusForbidden {
		t.Fatalf("execute without bisque.run_module status = %d body=%s, want 403", rec.Code, rec.Body.String())
	}

	run, lease := newRun(domain.RemoteMutationIntentBisqueRunModule)
	first := post(run, lease, "/v2/bisque/module-execute", execute)
	if first.Code != http.StatusOK {
		t.Fatalf("execute status = %d body=%s", first.Code, first.Body.String())
	}
	replay := post(run, lease, "/v2/bisque/module-execute", execute)
	if replay.Code != http.StatusOK || !strings.Contains(replay.Body.String(), "00-new") {
		t.Fatalf("replay status = %d body=%s", replay.Code, replay.Body.String())
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.executeCalls != 1 {
		t.Fatalf("execute reached BisQue %d times, want 1 (redelivery must replay the receipt)", state.executeCalls)
	}
	if state.executeUser != "ecologist" || state.executePass != "linked-secret" {
		t.Fatalf("worker execute used credentials %q, want the run-bound linked account", state.executeUser)
	}
}

func TestRunModuleIntentIsAcceptedAndWorkerScoped(t *testing.T) {
	t.Parallel()
	parsed, ok := domain.ParseRemoteMutationIntents([]string{"BISQUE.RUN_MODULE", "bisque.upload"})
	if !ok || len(parsed) != 2 || parsed[1] != domain.RemoteMutationIntentBisqueRunModule {
		t.Fatalf("ParseRemoteMutationIntents = %v %v", parsed, ok)
	}
	for _, path := range []string{"/v2/bisque/modules", "/v2/bisque/module", "/v2/bisque/module-execute"} {
		if !isWorkerBisqueEndpointPath(path) {
			t.Fatalf("%s must be reachable by run-bound workers", path)
		}
	}
}
