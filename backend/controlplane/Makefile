.PHONY: test run tidy generate generate-openapi generate-sqlc

generate: generate-openapi generate-sqlc

generate-openapi:
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen@v2.7.0 --config api/oapi-codegen.yaml api/openapi.yaml

generate-sqlc:
	go run github.com/sqlc-dev/sqlc/cmd/sqlc@v1.31.1 generate

test:
	go test ./...

run:
	go run ./cmd/ultra-control

tidy:
	go mod tidy
