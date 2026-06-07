package httpapi

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func BenchmarkResourceListingDataPlane(b *testing.B) {
	for _, count := range []int{100, 1000} {
		count := count
		b.Run(fmt.Sprintf("flat-scan-hash/%d-files", count), func(b *testing.B) {
			root, records := benchmarkUploadRoot(b, count, 4096)
			b.ReportAllocs()
			b.SetBytes(int64(count * 4096))
			b.ResetTimer()
			for range b.N {
				resources, err := listUploadResources(root)
				if err != nil {
					b.Fatal(err)
				}
				if len(resources) != len(records) {
					b.Fatalf("resources = %d, want %d", len(resources), len(records))
				}
			}
		})

		b.Run(fmt.Sprintf("catalog-store-only/%d-rows-page-200", count), func(b *testing.B) {
			root, records := benchmarkUploadRoot(b, count, 4096)
			mem := benchmarkCatalogStore(b, root, records)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				page, err := mem.ListResourcesForUser(context.Background(), domain.ResourceListInput{
					UserID: "bench-user",
					OrgID:  "bench-org",
					Status: "active",
					Limit:  200,
				})
				if err != nil {
					b.Fatal(err)
				}
				if len(page.Resources) != min(200, len(records)) {
					b.Fatalf("page resources = %d", len(page.Resources))
				}
			}
		})

		b.Run(fmt.Sprintf("catalog-page-shaping-stat/%d-rows-page-200", count), func(b *testing.B) {
			root, records := benchmarkUploadRoot(b, count, 4096)
			mem := benchmarkCatalogStore(b, root, records)
			deps := ServerDeps{Store: mem}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				page, err := mem.ListResourcesForUser(context.Background(), domain.ResourceListInput{
					UserID: "bench-user",
					OrgID:  "bench-org",
					Status: "active",
					Limit:  200,
				})
				if err != nil {
					b.Fatal(err)
				}
				shaped := make([]resourceRecord, 0, len(page.Resources))
				for _, resource := range page.Resources {
					shaped = append(shaped, deps.resourceRecordFromCatalog(root, resource))
				}
				if len(shaped) != min(200, len(records)) {
					b.Fatalf("shaped resources = %d", len(shaped))
				}
			}
		})
	}
}

func BenchmarkResourceReconcilerDataPlane(b *testing.B) {
	for _, count := range []int{100, 1000} {
		count := count
		b.Run(fmt.Sprintf("reconcile-clean/%d-files", count), func(b *testing.B) {
			root, records := benchmarkUploadRoot(b, count, 4096)
			mem := benchmarkCatalogStore(b, root, records)
			router := NewRouter(ServerDeps{
				Version:    "bench-version",
				Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
				Store:      mem,
				UploadRoot: root,
			})
			b.ReportAllocs()
			b.SetBytes(int64(count * 4096))
			b.ResetTimer()
			for range b.N {
				req := httptest.NewRequest(http.MethodPost, "/v2/admin/resources/reconcile", nil)
				req.Header.Set("X-Ultra-User-Id", "bench-user")
				req.Header.Set("X-Ultra-Org-Id", "bench-org")
				rec := httptest.NewRecorder()
				router.ServeHTTP(rec, req)
				if rec.Code != http.StatusOK {
					b.Fatalf("reconcile status = %d body=%s", rec.Code, rec.Body.String())
				}
			}
		})
	}
}

func BenchmarkExistingUploadRootScan(b *testing.B) {
	root := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_UPLOAD_ROOT_BENCH"))
	if root == "" {
		b.Skip("set ULTRA_CONTROL_UPLOAD_ROOT_BENCH to benchmark a real existing upload root")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		b.Fatal(err)
	}
	resources, err := listUploadResources(root)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportMetric(float64(len(resources)), "files")
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := listUploadResources(root); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExistingUploadRootRawFileHash(b *testing.B) {
	root := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_UPLOAD_ROOT_BENCH"))
	if root == "" {
		b.Skip("set ULTRA_CONTROL_UPLOAD_ROOT_BENCH to benchmark a real existing upload root")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		b.Fatal(err)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		b.Fatal(err)
	}
	files := make([]string, 0, len(entries))
	var totalBytes int64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		path := filepath.Join(root, entry.Name())
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			continue
		}
		files = append(files, path)
		totalBytes += info.Size()
	}
	if len(files) == 0 {
		b.Fatalf("no regular files found under %s", root)
	}
	b.ReportMetric(float64(len(files)), "files")
	b.ReportMetric(float64(totalBytes)/(1024*1024*1024), "GiB")
	b.ReportAllocs()
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for range b.N {
		for _, path := range files {
			if _, err := sha256File(path); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func benchmarkUploadRoot(b *testing.B, count int, size int) (string, []resourceRecord) {
	b.Helper()
	root := b.TempDir()
	payload := make([]byte, size)
	for index := range payload {
		payload[index] = byte(index % 251)
	}
	records := make([]resourceRecord, 0, count)
	baseTime := time.Now().UTC().Add(-time.Duration(count) * time.Second)
	for index := 0; index < count; index++ {
		fileID := fmt.Sprintf("file_bench_%06d", index)
		name := fmt.Sprintf("bench-%06d.bin", index)
		path := filepath.Join(root, fileID+"__"+name)
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			b.Fatalf("write benchmark upload: %v", err)
		}
		mtime := baseTime.Add(time.Duration(index) * time.Second)
		if err := os.Chtimes(path, mtime, mtime); err != nil {
			b.Fatalf("set benchmark upload mtime: %v", err)
		}
		if err := writeUploadMetadata(root, fileID, requestPrincipal{UserID: "bench-user", OrgID: "bench-org", Role: "researcher"}); err != nil {
			b.Fatalf("write benchmark metadata: %v", err)
		}
		record, err := uploadResourceFromPath(root, path)
		if err != nil {
			b.Fatalf("read benchmark upload resource: %v", err)
		}
		records = append(records, record)
	}
	return root, records
}

func benchmarkCatalogStore(b *testing.B, root string, records []resourceRecord) *store.MemoryStore {
	b.Helper()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	for _, record := range records {
		path := filepath.Join(root, record.FileID+"__"+record.OriginalName)
		_, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   record.FileID,
			OriginalName: record.OriginalName,
			ContentType:  record.ContentType,
			SizeBytes:    record.SizeBytes,
			SHA256:       record.SHA256,
			StorageURI:   fileStorageURI(path),
			StoragePath:  filepath.Base(path),
			SourceType:   record.SourceType,
			ResourceKind: record.ResourceKind,
			OwnerUserID:  "bench-user",
			OwnerOrgID:   "bench-org",
			OwnerRole:    "researcher",
			Status:       "active",
			CreatedAt:    parseResourceCreatedAt(record.CreatedAt),
			UpdatedAt:    domain.Now(),
		})
		if err != nil {
			b.Fatalf("upsert benchmark catalog resource: %v", err)
		}
	}
	return mem
}
