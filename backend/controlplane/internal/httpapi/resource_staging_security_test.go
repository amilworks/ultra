package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestCreateRunAuthorizesAndDeduplicatesSelectedResourcesBeforeDispatch(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    service,
		Store:   mem,
		WorkOS:  testWorkOSAuth(t, WorkOSAuthConfig{}),
	})
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "workos:user_a",
		Title:  "tenant staging",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	seedRunStagingResource(t, mem, domain.ResourceRecord{
		ResourceID: "file-owned", OwnerUserID: "workos:user_a", OwnerOrgID: "org-a",
		OriginalName: "owned.csv", ContentType: "text/csv",
		SizeBytes: 21274, SHA256: strings.Repeat("c", 64),
		Metadata: domain.JSONMap{
			"source":      "upload_store",
			"caption":     "owner-declared measurements",
			"credentials": domain.JSONMap{"token": "descriptor-secret"},
		},
	})
	seedRunStagingResource(t, mem, domain.ResourceRecord{
		ResourceID: "file-foreign", OwnerUserID: "user-b", OwnerOrgID: "org-b",
		OriginalName: "private.csv", ContentType: "text/csv",
	})
	cookie := testWorkOSSessionCookie(t, "user_a", "user-a@example.org", "org-a", "researcher")

	post := func(body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.AddCookie(cookie)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	// A foreign/unreadable selected id is DROPPED (not a 404) so a stale client
	// selection cannot block run creation. The run proceeds with only the
	// readable id, and the response still masks whether the foreign id exists.
	dropped := post(`{"user_id":"body-forged-user","goal":"load resources","file_ids":["file-owned","file-foreign"]}`)
	if dropped.Code != http.StatusOK {
		t.Fatalf("mixed selection status = %d body=%s, want 200 (foreign id dropped)", dropped.Code, dropped.Body.String())
	}
	if strings.Contains(dropped.Body.String(), "file-foreign") || strings.Contains(dropped.Body.String(), "user-b") {
		t.Fatalf("foreign resource existence leaked: %s", dropped.Body.String())
	}
	var droppedRun domain.RunRecord
	if err := json.Unmarshal(dropped.Body.Bytes(), &droppedRun); err != nil {
		t.Fatalf("decode dropped-selection run: %v", err)
	}
	if got := droppedRun.Metadata["file_ids"]; !jsonArrayEquals(got, []string{"file-owned"}) {
		t.Fatalf("dropped-selection run file_ids = %#v, want only the readable id", droppedRun.Metadata["file_ids"])
	}
	select {
	case job := <-bus.Jobs():
		if len(job.FileIDs) != 1 || job.FileIDs[0] != "file-owned" {
			t.Fatalf("dropped-selection job file_ids = %#v, want only [file-owned]", job.FileIDs)
		}
	default:
		t.Fatal("dropped-selection did not dispatch a job")
	}

	accepted := post(`{"user_id":"body-forged-user","goal":"load resources","file_ids":[" file-owned ","file-owned","file-owned"],"resource_descriptors":[{"type":"selected_resource","resource_id":"file-owned","file_id":"file-owned","sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","size_bytes":1,"metadata":{"caption":"forged","credentials":{"token":"body-descriptor-secret"}}}]}`)
	if accepted.Code != http.StatusOK {
		t.Fatalf("owned selected id status = %d body=%s, want 200", accepted.Code, accepted.Body.String())
	}
	if strings.Contains(accepted.Body.String(), "body-descriptor-secret") || strings.Contains(accepted.Body.String(), strings.Repeat("f", 64)) {
		t.Fatalf("caller-selected descriptor leaked into run response: %s", accepted.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(accepted.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.UserID != "workos:user_a" || run.Metadata["org_id"] != "org-a" {
		t.Fatalf("run principal metadata = user %q metadata %#v", run.UserID, run.Metadata)
	}
	if got := run.Metadata["file_ids"]; !jsonArrayEquals(got, []string{"file-owned"}) {
		t.Fatalf("deduplicated run metadata file_ids = %#v", run.Metadata["file_ids"])
	}
	select {
	case job := <-bus.Jobs():
		if len(job.FileIDs) != 1 || job.FileIDs[0] != "file-owned" {
			t.Fatalf("deduplicated job file_ids = %#v", job.FileIDs)
		}
		assertSelectedResourceBinding(t, job.ResourceDescriptors, "file-owned")
	default:
		t.Fatal("owned selection did not dispatch a job")
	}
	stored, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	descriptors, ok := stored.Metadata["resource_descriptors"].([]domain.JSONMap)
	if !ok {
		t.Fatalf("stored selected resource descriptors = %T %#v", stored.Metadata["resource_descriptors"], stored.Metadata["resource_descriptors"])
	}
	assertSelectedResourceBinding(t, descriptors, "file-owned")

	forgedMetadata := post(`{"goal":"forge selected capability","metadata":{"file_ids":["file-owned"],"resource_descriptors":[{"type":"selected_resource","resource_id":"file-owned"}]}}`)
	if forgedMetadata.Code != http.StatusOK {
		t.Fatalf("reserved metadata status = %d body=%s, want 200", forgedMetadata.Code, forgedMetadata.Body.String())
	}
	var forgedRun domain.RunRecord
	if err := json.Unmarshal(forgedMetadata.Body.Bytes(), &forgedRun); err != nil {
		t.Fatalf("decode forged metadata run: %v", err)
	}
	if _, exists := forgedRun.Metadata["file_ids"]; exists {
		t.Fatalf("caller-forged file_ids survived in run metadata: %#v", forgedRun.Metadata)
	}
	if _, exists := forgedRun.Metadata["resource_descriptors"]; exists {
		t.Fatalf("caller-forged resource descriptors survived in run metadata: %#v", forgedRun.Metadata)
	}
	select {
	case job := <-bus.Jobs():
		if len(job.FileIDs) != 0 || len(job.ResourceDescriptors) != 0 {
			t.Fatalf("caller-forged resource capability reached worker job: %+v", job)
		}
	default:
		t.Fatal("forged metadata run did not dispatch a job")
	}
}

func TestCreateRunDropsCallerAuthoredArtifactDescriptorsBeforeDispatch(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    service,
		Store:   mem,
		WorkOS:  testWorkOSAuth(t, WorkOSAuthConfig{}),
	})

	foreignThread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "workos:user_b",
		Title:  "Foreign artifacts",
	})
	if err != nil {
		t.Fatalf("CreateThread foreign: %v", err)
	}
	foreignRun, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: foreignThread.ThreadID,
		UserID:   "workos:user_b",
		Goal:     "Create a private artifact.",
	})
	if err != nil {
		t.Fatalf("CreateRun foreign: %v", err)
	}
	drainJobs(bus)
	foreignArtifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		ArtifactID: "artifact-foreign-secret",
		RunID:      foreignRun.RunID,
		ThreadID:   foreignThread.ThreadID,
		Kind:       "table",
		Path:       "outputs/private.csv",
		SourcePath: "/srv/ultra/artifacts/" + foreignRun.RunID + "/outputs/private.csv",
		Title:      "Private foreign table",
	})
	if err != nil {
		t.Fatalf("CreateArtifact foreign: %v", err)
	}

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "workos:user_a",
		Title:  "Artifact descriptor boundary",
	})
	if err != nil {
		t.Fatalf("CreateThread attacker: %v", err)
	}
	body, err := json.Marshal(map[string]any{
		"goal": "stage a guessed foreign artifact",
		"resource_descriptors": []map[string]any{
			{
				"type":        "artifact",
				"artifact_id": foreignArtifact.ArtifactID,
				"run_id":      foreignRun.RunID,
				"path":        foreignArtifact.Path,
				"source_path": foreignArtifact.SourcePath,
			},
			{
				// The runtime historically defaulted a missing type to "artifact".
				"artifact_id": "artifact-foreign-untyped",
				"run_id":      foreignRun.RunID,
				"path":        foreignArtifact.Path,
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(
		http.MethodPost,
		"/v2/threads/"+thread.ThreadID+"/runs",
		strings.NewReader(string(body)),
	)
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(testWorkOSSessionCookie(
		t, "user_a", "user-a@example.org", "org-a", "researcher",
	))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), foreignArtifact.ArtifactID) ||
		strings.Contains(rec.Body.String(), foreignRun.RunID) ||
		strings.Contains(rec.Body.String(), foreignArtifact.SourcePath) {
		t.Fatalf("foreign artifact capability leaked into run response: %s", rec.Body.String())
	}

	var created domain.RunRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if _, exists := created.Metadata["resource_descriptors"]; exists {
		t.Fatalf("caller artifact descriptors survived in run metadata: %#v", created.Metadata)
	}
	select {
	case job := <-bus.Jobs():
		if len(job.ResourceDescriptors) != 0 {
			t.Fatalf("caller artifact descriptors reached worker job: %#v", job.ResourceDescriptors)
		}
	default:
		t.Fatal("create run did not dispatch a job")
	}
	stored, err := mem.GetRun(ctx, created.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if _, exists := stored.Metadata["resource_descriptors"]; exists {
		t.Fatalf("caller artifact descriptors persisted: %#v", stored.Metadata)
	}
}

func TestSelectedAndResolvedResourceProjectionsCarryOnlyCatalogBoundTreeIdentity(t *testing.T) {
	root := t.TempDir()
	resourceID := "file-sensor-tree"
	dir := filepath.Join(root, bundlesDirName, resourceID, "signals.zarr")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(dir, ".zattrs"),
		[]byte(`{"ultra":{"sensor_series":{"schema":"ultra.sensor-series.v1"}}}`),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("a", 64)
	resource := domain.ResourceRecord{
		ResourceID:   resourceID,
		OwnerUserID:  "owner",
		OriginalName: "renamed-display-name",
		SizeBytes:    4096,
		SHA256:       digest,
		StoragePath:  dir,
		StorageURI:   fileStorageURI(dir),
		SourceType:   "upload",
		ResourceKind: "image",
	}

	descriptors := withAuthorizedSelectedResourceDescriptors(
		nil,
		[]domain.ResourceRecord{resource},
	)
	tree, ok := descriptors[0]["tree_identity"].(domain.JSONMap)
	if !ok {
		t.Fatalf("selected tree identity=%T %#v", descriptors[0]["tree_identity"], descriptors[0])
	}
	if tree["tree_manifest_sha256"] != digest || tree["tree_manifest_path"] != treeManifestPath ||
		tree["authority"] != "control_resource_catalog" {
		t.Fatalf("selected tree identity=%#v", tree)
	}
	sensor, ok := descriptors[0]["sensor_format"].(domain.JSONMap)
	if !ok || sensor["schema"] != sensorFormatBindingSchema ||
		sensor["sensor_schema"] != sensorSeriesSchema || sensor["resource_sha256"] != digest ||
		sensor["authority"] != "control_resource_catalog" {
		t.Fatalf("selected sensor format=%T %#v", descriptors[0]["sensor_format"], descriptors[0])
	}
	resolved := runResourceHitFromRecord(resource, "")
	if resolved.TreeIdentity["tree_manifest_sha256"] != digest {
		t.Fatalf("resolved tree identity=%#v", resolved.TreeIdentity)
	}
	if resolved.SensorFormat["resource_sha256"] != digest {
		t.Fatalf("resolved sensor format=%#v", resolved.SensorFormat)
	}

	// A caller/catalog row cannot turn an ordinary file or a symlink-shaped
	// path into a directory identity merely by supplying a SHA-256 value.
	ordinary := filepath.Join(root, "file-ordinary__signals.zarr")
	if err := os.WriteFile(ordinary, []byte("not a tree"), 0o644); err != nil {
		t.Fatal(err)
	}
	resource.ResourceID = "file-ordinary"
	resource.StoragePath = ordinary
	resource.StorageURI = fileStorageURI(ordinary)
	if tree := projectCatalogTreeIdentity(resource); len(tree) != 0 {
		t.Fatalf("ordinary file received tree identity: %#v", tree)
	}
	if sensor := projectCatalogSensorFormat(resource); len(sensor) != 0 {
		t.Fatalf("ordinary file received sensor format: %#v", sensor)
	}
	symlink := filepath.Join(root, bundlesDirName, "file-linked", "signals.zarr")
	if err := os.MkdirAll(filepath.Dir(symlink), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(dir, symlink); err != nil {
		t.Fatal(err)
	}
	resource.ResourceID = "file-linked"
	resource.StoragePath = symlink
	resource.StorageURI = fileStorageURI(symlink)
	if tree := projectCatalogTreeIdentity(resource); len(tree) != 0 {
		t.Fatalf("symlink received tree identity: %#v", tree)
	}
	if sensor := projectCatalogSensorFormat(resource); len(sensor) != 0 {
		t.Fatalf("symlink received sensor format: %#v", sensor)
	}
}

func TestCatalogSensorFormatRejectsGenericNGFFMalformedAndOversizedAttributes(t *testing.T) {
	root := t.TempDir()
	digest := strings.Repeat("b", 64)
	resource := domain.ResourceRecord{
		OwnerUserID: "owner", OriginalName: "dataset.ome.zarr", SHA256: digest,
		SourceType: "upload", ResourceKind: "image",
	}
	writeBundle := func(resourceID string, name string, payload []byte) string {
		t.Helper()
		dir := filepath.Join(root, bundlesDirName, resourceID, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, ".zattrs"), payload, 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}

	for _, fixture := range []struct {
		id      string
		payload []byte
	}{
		{id: "biology-ngff", payload: []byte(`{"multiscales":[],"omero":{"channels":[]}}`)},
		{id: "wrong-schema", payload: []byte(`{"ultra":{"sensor_series":{"schema":"other"}}}`)},
		{id: "malformed", payload: []byte(`{"ultra":`)},
		{id: "oversized", payload: make([]byte, maxSensorRootAttributesBytes+1)},
	} {
		resource.ResourceID = fixture.id
		resource.StoragePath = writeBundle(fixture.id, resource.OriginalName, fixture.payload)
		resource.StorageURI = fileStorageURI(resource.StoragePath)
		if marker := projectCatalogSensorFormat(resource); len(marker) != 0 {
			t.Fatalf("fixture %s received sensor marker: %#v", fixture.id, marker)
		}
	}
}

func TestCatalogSensorFormatDetectsBoundedZarrV3Attributes(t *testing.T) {
	root := t.TempDir()
	resourceID := "sensor-v3"
	dir := filepath.Join(root, bundlesDirName, resourceID, "signals.zarr")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(dir, "zarr.json"),
		[]byte(`{"zarr_format":3,"node_type":"group","attributes":{"ultra.sensor_series":{"schema":"ultra.sensor-series.v1"}}}`),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("c", 64)
	resource := domain.ResourceRecord{
		ResourceID: resourceID, OriginalName: "signals.zarr", SHA256: digest,
		StoragePath: dir, StorageURI: fileStorageURI(dir), SourceType: "upload",
	}
	marker := projectCatalogSensorFormat(resource)
	if marker["sensor_schema"] != sensorSeriesSchema || marker["resource_sha256"] != digest {
		t.Fatalf("zarr v3 sensor marker=%#v", marker)
	}
}

func TestRunResourceEndpointsIgnoreForgedOrgHeaderAndUseStampedRunOrg(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		UploadRoot:  t.TempDir(),
		WorkerToken: "worker-secret",
	})
	seedRunStagingResource(t, mem, domain.ResourceRecord{
		ResourceID: "file-trusted-org", OwnerUserID: "user-a", OwnerOrgID: "org-trusted",
		OriginalName: "trusted.csv", ContentType: "text/csv",
	})
	seedRunStagingResource(t, mem, domain.ResourceRecord{
		ResourceID: "file-forged-org", OwnerUserID: "user-a", OwnerOrgID: "org-forged",
		OriginalName: "forged-scope.csv", ContentType: "text/csv",
	})
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-a", Title: "trusted org"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-a",
		Goal:     "resolve a resource",
		Metadata: domain.JSONMap{
			"org_id": "org-trusted",
			"principal": domain.JSONMap{
				"user_id": "user-a",
				"org_id":  "org-trusted",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	post := func(path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Ultra-Worker-Token", "worker-secret")
		req.Header.Set("X-Ultra-Org-Id", "org-forged")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	var search struct {
		Resources []runResourceHit `json:"resources"`
	}
	searchRec := post("/v2/runs/"+run.RunID+"/resource-search", `{"limit":50}`)
	if searchRec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s", searchRec.Code, searchRec.Body.String())
	}
	if err := json.Unmarshal(searchRec.Body.Bytes(), &search); err != nil {
		t.Fatalf("decode search: %v", err)
	}
	if len(search.Resources) != 1 || search.Resources[0].ResourceID != "file-trusted-org" {
		t.Fatalf("forged org widened search: %+v", search.Resources)
	}

	var resolved runResourceResolveResponse
	resolveRec := post(
		"/v2/runs/"+run.RunID+"/resource-resolve",
		`{"resource_ids":["file-trusted-org","file-forged-org"]}`,
	)
	if resolveRec.Code != http.StatusOK {
		t.Fatalf("resolve status = %d body=%s", resolveRec.Code, resolveRec.Body.String())
	}
	if err := json.Unmarshal(resolveRec.Body.Bytes(), &resolved); err != nil {
		t.Fatalf("decode resolve: %v", err)
	}
	if len(resolved.Resources) != 1 || resolved.Resources[0].ResourceID != "file-trusted-org" {
		t.Fatalf("forged org widened resolve: %+v", resolved.Resources)
	}
	if len(resolved.Missing) != 1 || resolved.Missing[0] != "file-forged-org" {
		t.Fatalf("resolve missing = %#v", resolved.Missing)
	}
}

// The model-visible projection is a deny-by-default allowlist: safe generic
// keys survive, and secrets, license bodies and unknown nested objects (such as
// the former CALPHAD provenance block) never cross the model boundary.
func TestRunResourceMetadataProjectionRedactsSecretsAndDeniesUnknownKeys(t *testing.T) {
	hit := runResourceHitFromRecord(domain.ResourceRecord{
		ResourceID:   "file-measurements",
		OriginalName: "measurements.csv",
		SizeBytes:    21274,
		SHA256:       strings.Repeat("b", 64),
		Metadata: domain.JSONMap{
			"source":                 "upload_store",
			"caption":                "assessed measurements",
			"scientific_descriptors": []string{"tabular measurements"},
			"api_key":                "top-level-secret",
			"license_text":           "private legal terms",
			"credentials":            domain.JSONMap{"password": "credential-secret"},
			"calphad": domain.JSONMap{
				"database_id": "nist-al-co-w-wang-2017",
				"credentials": domain.JSONMap{"token": "vendor-secret"},
			},
			"vendor_private_terms": "forbidden vendor terms",
		},
	}, "")

	if hit.Metadata["source"] != "upload_store" || hit.Metadata["caption"] != "assessed measurements" {
		t.Fatalf("safe generic metadata missing: %#v", hit.Metadata)
	}
	if _, exists := hit.Metadata["scientific_descriptors"]; !exists {
		t.Fatalf("safe scientific_descriptors missing: %#v", hit.Metadata)
	}
	for _, key := range []string{"calphad", "api_key", "license_text", "credentials", "vendor_private_terms"} {
		if _, exists := hit.Metadata[key]; exists {
			t.Fatalf("denied metadata key %q leaked: %#v", key, hit.Metadata)
		}
	}
	if hit.SHA256 != strings.Repeat("b", 64) || hit.SizeBytes != 21274 {
		t.Fatalf("immutable catalog binding = sha %q size %d", hit.SHA256, hit.SizeBytes)
	}
	encoded, err := json.Marshal(hit)
	if err != nil {
		t.Fatalf("marshal hit: %v", err)
	}
	for _, secret := range []string{
		"top-level-secret", "private legal terms", "credential-secret",
		"forbidden vendor terms", "vendor-secret", "nist-al-co-w-wang-2017",
	} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("secret %q leaked in model projection: %s", secret, encoded)
		}
	}
}

// A source declaration that smuggles credentials or license terms is rejected
// rather than projected.
func TestRunResourceMetadataProjectionRejectsUnsafeOwnerDeclarations(t *testing.T) {
	for _, unsafe := range []string{
		"https://vendor-user:credential-secret@example.org/private",
		"Confidential proprietary license agreement",
		"Private license text: do not distribute",
	} {
		hit := runResourceHitFromRecord(domain.ResourceRecord{
			ResourceID: "file-decl",
			Metadata:   domain.JSONMap{"source": unsafe},
		}, "")
		if got, exists := hit.Metadata["source"]; exists {
			t.Fatalf("unsafe owner declaration %q projected as %#v", unsafe, got)
		}
	}
	safe := runResourceHitFromRecord(domain.ResourceRecord{
		ResourceID: "file-decl-safe",
		Metadata:   domain.JSONMap{"source": "https://example.org/public/dataset"},
	}, "")
	if safe.Metadata["source"] != "https://example.org/public/dataset" {
		t.Fatalf("safe owner declaration dropped: %#v", safe.Metadata)
	}
}

func seedRunStagingResource(t *testing.T, mem *store.MemoryStore, resource domain.ResourceRecord) {
	t.Helper()
	now := domain.Now()
	sizeBytes := resource.SizeBytes
	if sizeBytes <= 0 {
		sizeBytes = 1024
	}
	sha256 := resource.SHA256
	if sha256 == "" {
		sha256 = strings.Repeat("b", 64)
	}
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   resource.ResourceID,
		OwnerUserID:  resource.OwnerUserID,
		OwnerOrgID:   resource.OwnerOrgID,
		OwnerRole:    "researcher",
		OriginalName: resource.OriginalName,
		ContentType:  resource.ContentType,
		SizeBytes:    sizeBytes,
		SHA256:       sha256,
		StorageURI:   "file:///private/" + resource.ResourceID,
		StoragePath:  "/private/" + resource.ResourceID,
		SourceType:   "upload",
		ResourceKind: "document",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     resource.Metadata,
	}); err != nil {
		t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
	}
}

func assertSelectedResourceBinding(t *testing.T, descriptors []domain.JSONMap, resourceID string) {
	t.Helper()
	for _, descriptor := range descriptors {
		if descriptor["type"] != "selected_resource" || descriptor["resource_id"] != resourceID {
			continue
		}
		if descriptor["binding_schema"] != "ultra.selected_resource.v1" || descriptor["authority"] != "control_resource_catalog" {
			t.Fatalf("binding authority = %#v", descriptor)
		}
		if descriptor["file_id"] != resourceID || descriptor["original_name"] != "owned.csv" ||
			descriptor["content_type"] != "text/csv" {
			t.Fatalf("selected resource identity = %#v", descriptor)
		}
		if descriptor["sha256"] != strings.Repeat("c", 64) || descriptor["size_bytes"] != int64(21274) {
			t.Fatalf("selected resource hash/size binding = %#v", descriptor)
		}
		metadata, ok := descriptor["metadata"].(domain.JSONMap)
		if !ok {
			t.Fatalf("selected resource metadata = %T %#v", descriptor["metadata"], descriptor["metadata"])
		}
		// The catalog row is the only metadata authority, and it is projected
		// through the deny-by-default allowlist.
		if metadata["source"] != "upload_store" || metadata["caption"] != "owner-declared measurements" {
			t.Fatalf("selected resource metadata = %#v", metadata)
		}
		if _, exists := metadata["credentials"]; exists {
			t.Fatalf("catalog credentials leaked into descriptor: %#v", metadata)
		}
		encoded, _ := json.Marshal(descriptor)
		if strings.Contains(string(encoded), "descriptor-secret") || strings.Contains(string(encoded), "body-descriptor-secret") || strings.Contains(string(encoded), strings.Repeat("f", 64)) {
			t.Fatalf("forged selected metadata leaked: %s", encoded)
		}
		return
	}
	t.Fatalf("missing selected resource descriptor for %q: %#v", resourceID, descriptors)
}

// A remote-mutation intent from the client, with no linked BisQue account, must
// NOT fail the chat turn at run creation. The run starts without the mutation
// capability; the mutating tool then fails gracefully at tool time.
func TestCreateRunWithoutBisqueAccountDropsMutationIntentInsteadOf409(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    service,
		Store:   mem,
		WorkOS:  testWorkOSAuth(t, WorkOSAuthConfig{}),
	})
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "workos:user_a",
		Title:  "bisque intent without account",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	cookie := testWorkOSSessionCookie(t, "user_a", "user-a@example.org", "org-a", "researcher")

	req := httptest.NewRequest(
		http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs",
		strings.NewReader(`{"goal":"upload the plot to my BisQue account","remote_mutation_intents":["bisque.upload"]}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(cookie)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("run-create status = %d body=%s, want 200 (intent dropped, turn not failed)", rec.Code, rec.Body.String())
	}
	select {
	case job := <-bus.Jobs():
		if len(job.RemoteMutationIntents) != 0 {
			t.Fatalf("run started with mutation intents despite no linked account: %#v", job.RemoteMutationIntents)
		}
	default:
		t.Fatal("run did not dispatch a job")
	}
}
