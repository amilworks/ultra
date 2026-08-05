package openapi

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/go-chi/chi/v5"
)

type uploadTileCaptureHandler struct {
	Unimplemented
	params GetUploadTileParams
}

func (handler *uploadTileCaptureHandler) GetUploadTile(
	w http.ResponseWriter,
	_ *http.Request,
	_ FileID,
	_ GetUploadTileParamsAxis,
	_ int,
	_ int,
	_ int,
	params GetUploadTileParams,
) {
	handler.params = params
	w.WriteHeader(http.StatusNoContent)
}

func TestGetUploadTileWrapperBindsChannelIdentity(t *testing.T) {
	t.Parallel()

	handler := &uploadTileCaptureHandler{}
	router := chi.NewRouter()
	wrapped := HandlerFromMux(handler, router)
	request := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/file-1/tiles/z/2/3/4?size=1024&channels=5%2C1%2C259&channel_colors=%230000ff%2C%2300ff00%2C%23ff0000&cache_key=channels-v1",
		nil,
	)
	recorder := httptest.NewRecorder()

	wrapped.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	if handler.params.Size == nil || *handler.params.Size != 1024 {
		t.Fatalf("size = %v, want 1024", handler.params.Size)
	}
	if handler.params.Channels == nil || *handler.params.Channels != "5,1,259" {
		t.Fatalf("channels = %v", handler.params.Channels)
	}
	if handler.params.ChannelColors == nil || *handler.params.ChannelColors != "#0000ff,#00ff00,#ff0000" {
		t.Fatalf("channel_colors = %v", handler.params.ChannelColors)
	}
	if handler.params.CacheKey == nil || *handler.params.CacheKey != "channels-v1" {
		t.Fatalf("cache_key = %v", handler.params.CacheKey)
	}
}

func TestGeneratedImageChannelContractMatchesOpenAPI(t *testing.T) {
	t.Parallel()

	swagger, err := GetSwagger()
	if err != nil {
		t.Fatalf("GetSwagger: %v", err)
	}
	for path, requiredParams := range map[string][]string{
		"/v2/uploads/{file_id}/slice":                                  {"axis", "x", "y", "z", "level", "c", "channel", "channels", "channel_colors", "cache_key", "t", "time", "timepoint"},
		"/v2/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}": {"size", "c", "channel", "channels", "channel_colors", "cache_key"},
		"/v2/uploads/{file_id}/atlas":                                  {"t", "level", "grid_rows", "grid_cols", "scale", "c", "channel", "channels", "channel_colors"},
	} {
		operation := swagger.Paths.Find(path).Get
		for _, name := range requiredParams {
			if operation.Parameters.GetByInAndName("query", name) == nil {
				t.Errorf("%s missing generated query parameter %q", path, name)
			}
		}
		if operation.Responses.Value("422") == nil {
			t.Errorf("%s missing generated 422 response", path)
		}
		if operation.Responses.Value("404") == nil {
			t.Errorf("%s missing generated authorization-safe 404 response", path)
		}
	}
	for _, path := range []string{
		"/v2/uploads/{file_id}/slice",
		"/v2/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}",
		"/v2/uploads/{file_id}/atlas",
	} {
		response := swagger.Paths.Find(path).Get.Responses.Value("200")
		if response == nil || response.Value == nil || response.Value.Content.Get("image/png") == nil {
			t.Errorf("%s generated 200 response is not image/png", path)
		}
	}

	tileOperation := swagger.Paths.Find("/v2/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}").Get
	tileAxis := tileOperation.Parameters.GetByInAndName("path", "axis").Schema.Value
	if len(tileAxis.Enum) != 1 || tileAxis.Enum[0] != "z" {
		t.Fatalf("generated tile axis enum = %v, want [z]", tileAxis.Enum)
	}
	sizeSchema := tileOperation.Parameters.GetByInAndName("query", "size").Schema.Value
	if sizeSchema.Max == nil || *sizeSchema.Max != 1024 {
		t.Fatalf("generated tile size maximum = %v, want 1024", sizeSchema.Max)
	}

	atlasType := reflect.TypeOf(GetUploadAtlasParams{})
	if atlasType.NumField() != 9 {
		t.Fatalf("GetUploadAtlasParams fields = %d, want t/level/grid/scale/channel identity", atlasType.NumField())
	}
}

func TestGeneratedImageAliasesMediaAndStatusTruth(t *testing.T) {
	t.Parallel()

	swagger, err := GetSwagger()
	if err != nil {
		t.Fatalf("GetSwagger: %v", err)
	}
	for path, aliases := range map[string][]string{
		"/v2/uploads/{file_id}/scalar-volume": {"c", "channel", "t", "time", "timepoint"},
		"/v2/uploads/{file_id}/histogram":     {"c", "channel", "channels", "t", "time", "timepoint"},
	} {
		operation := swagger.Paths.Find(path).Get
		for _, alias := range aliases {
			if operation.Parameters.GetByInAndName("query", alias) == nil {
				t.Errorf("%s missing generated selector alias %q", path, alias)
			}
		}
	}

	for path, mediaTypes := range map[string][]string{
		"/v2/uploads/{file_id}/preview":     {"image/png", "image/jpeg"},
		"/v2/uploads/{file_id}/display":     {"image/png", "image/jpeg"},
		"/v2/uploads/{file_id}/slice":       {"image/png", "image/jpeg"},
		"/v2/resources/{file_id}/thumbnail": {"image/png"},
	} {
		response := swagger.Paths.Find(path).Get.Responses.Value("200")
		for _, mediaType := range mediaTypes {
			if response == nil || response.Value == nil || response.Value.Content.Get(mediaType) == nil {
				t.Errorf("%s generated 200 response is missing %s", path, mediaType)
			}
		}
		for _, status := range []string{"404", "422"} {
			if swagger.Paths.Find(path).Get.Responses.Value(status) == nil {
				t.Errorf("%s missing generated %s response", path, status)
			}
		}
	}

	for path, statuses := range map[string][]string{
		"/v2/uploads/{file_id}/scalar-volume": {"400", "404", "415", "422", "502", "503"},
		"/v2/uploads/{file_id}/histogram":     {"400", "404", "415", "422", "502"},
	} {
		for _, status := range statuses {
			if swagger.Paths.Find(path).Get.Responses.Value(status) == nil {
				t.Errorf("%s missing generated truthful status %s", path, status)
			}
		}
	}
	if swagger.Paths.Find("/v2/uploads/{file_id}/scalar-volume").Get.Responses.Value("501") != nil {
		t.Error("scalar-volume advertises unreachable 501 response")
	}
	for path, status := range map[string]string{
		"/v2/uploads/{file_id}/slice":  "400",
		"/v2/uploads/{file_id}/viewer": "503",
	} {
		if swagger.Paths.Find(path).Get.Responses.Value(status) == nil {
			t.Errorf("%s missing generated truthful status %s", path, status)
		}
	}
	if swagger.Paths.Find("/v2/uploads/{file_id}/slice").Get.Responses.Value("500") == nil {
		t.Error("slice generated responses omit runtime upstream 500")
	}
	for _, path := range []string{
		"/v2/uploads/{file_id}/preview",
		"/v2/uploads/{file_id}/display",
	} {
		if swagger.Paths.Find(path).Get.Responses.Value("206") == nil {
			t.Errorf("%s missing generated range response 206", path)
		}
		for _, status := range []string{"200", "206"} {
			response := swagger.Paths.Find(path).Get.Responses.Value(status)
			if response == nil || response.Value == nil || response.Value.Content.Get("*/*") == nil {
				t.Errorf("%s generated %s response is missing wildcard binary media coverage", path, status)
			}
		}
	}
	scalarResponse := swagger.Paths.Find("/v2/uploads/{file_id}/scalar-volume").Get.Responses.Value("200")
	if scalarResponse == nil || scalarResponse.Value == nil || scalarResponse.Value.Headers["x-volume-time-count"] == nil {
		t.Error("scalar-volume generated 200 response omits required x-volume-time-count header")
	}
	if swagger.Paths.Find("/v2/uploads/{file_id}/display").Get.Parameters.GetByInAndName("query", "cache_key") == nil {
		t.Error("display generated parameters omit runtime cache_key")
	}
}

func TestGeneratedPyramidDerivationContract(t *testing.T) {
	t.Parallel()

	swagger, err := GetSwagger()
	if err != nil {
		t.Fatalf("GetSwagger: %v", err)
	}
	item := swagger.Paths.Find("/v2/uploads/{file_id}/derive-pyramid")
	if item == nil || item.Post == nil {
		t.Fatal("generated OpenAPI is missing POST /v2/uploads/{file_id}/derive-pyramid")
	}
	response := item.Post.Responses.Value("202")
	if response == nil || response.Value == nil || response.Value.Content.Get("application/json") == nil {
		t.Fatal("derive-pyramid generated 202 response is missing application/json")
	}
	for _, status := range []string{"422", "502"} {
		if item.Post.Responses.Value(status) == nil {
			t.Fatalf("derive-pyramid generated responses omit runtime %s", status)
		}
	}
	notConfigured := item.Post.Responses.Value("501")
	if notConfigured == nil || notConfigured.Value == nil {
		t.Fatal("derive-pyramid generated responses omit 501")
	}
	schema := notConfigured.Value.Content.Get("application/json")
	if schema == nil || schema.Schema == nil || schema.Schema.Ref != "#/components/schemas/V2NotConfiguredResponse" {
		t.Fatalf("derive-pyramid 501 schema = %#v, want V2NotConfiguredResponse", schema)
	}
}
