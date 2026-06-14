package httpapi

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// BenchmarkUploadCompleteReassembly measures the /complete reassembly throughput (the
// hot path that streams every chunk into one file + hashes it). Chunk upload is excluded
// from the timer; only the reassembly is measured. SetBytes reports MB/s.
func BenchmarkUploadCompleteReassembly(b *testing.B) {
	const chunkSize = 8 << 20 // 8 MiB, matching the frontend default
	type chunk struct {
		idx  int
		off  int64
		data []byte
		sha  string
	}
	for _, sizeMB := range []int{8, 128} {
		b.Run(fmt.Sprintf("%dMB", sizeMB), func(b *testing.B) {
			total := int64(sizeMB) << 20
			payload := make([]byte, total)
			for i := range payload {
				payload[i] = byte(i*131 + 7)
			}
			fileSum := sha256.Sum256(payload)
			fileSHA := hex.EncodeToString(fileSum[:])
			var chunks []chunk
			for off := int64(0); off < total; off += chunkSize {
				end := off + chunkSize
				if end > total {
					end = total
				}
				data := payload[off:end]
				sum := sha256.Sum256(data)
				chunks = append(chunks, chunk{idx: len(chunks), off: off, data: data, sha: hex.EncodeToString(sum[:])})
			}
			body := uploadSessionCreateBody("bench-reassembly", "f1", "big.bin", "application/octet-stream", payload, fileSHA)

			b.SetBytes(total)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				root := b.TempDir()
				mem := store.NewMemoryStore()
				router := NewRouter(ServerDeps{
					Version:    "bench",
					Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
					Store:      mem,
					UploadRoot: root,
				})
				created := createUploadSessionForBench(b, router, body, "u", "o", http.StatusCreated)
				for _, c := range chunks {
					uploadChunkForBench(b, router, created.Session.SessionID, "f1", c.idx, c.off, c.data, c.sha, "u", "o")
				}
				b.StartTimer()
				completeUploadSessionFileForBench(b, router, created.Session.SessionID, "f1", "u", "o", http.StatusOK)
			}
		})
	}
}
