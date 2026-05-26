.PHONY: test run tidy generate

generate:
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen --config api/oapi-codegen.yaml api/openapi.yaml

test:
	go test ./...

run:
	go run ./cmd/ultra-control

tidy:
	go mod tidy
