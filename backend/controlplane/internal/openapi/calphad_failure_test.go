package openapi

import (
	"os"
	"strings"
	"testing"
)

func TestOpenAPICalphadTerminalFailureContractIsGenerated(t *testing.T) {
	t.Parallel()
	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	for _, required := range []string{
		"enum: [pending, input_validated, equilibrium_completed, scheil_completed, failed, timeout, unsupported]",
		"failure_domain:",
		"enum: [input, scientific, platform]",
		"failure_stage:",
		"enum: [parse, solver, result_validation, sandbox_runtime]",
		"calphad_sandbox_timeout",
		"required: [failure_domain, failure_stage, failure_code]",
		"False for every terminal failure",
		"inspect terminal failure may omit it",
	} {
		if !strings.Contains(string(document), required) {
			t.Errorf("CALPHAD terminal failure contract missing %q", required)
		}
	}
	if !V2CalphadValidationAppendRequestStatusTimeout.Valid() ||
		!V2CalphadValidationAppendRequestStatusScheilCompleted.Valid() ||
		!V2CalphadValidationAppendRequestOperationScheil.Valid() ||
		!V2CalphadValidationRecordStatusTimeout.Valid() ||
		!V2CalphadValidationRecordStatusScheilCompleted.Valid() ||
		!V2CalphadValidationRecordOperationScheil.Valid() ||
		!V2CalphadValidationAppendRequestFailureDomainPlatform.Valid() ||
		!V2CalphadValidationAppendRequestFailureStageSandboxRuntime.Valid() ||
		!V2CalphadValidationAppendRequestFailureCodeCalphadSandboxTimeout.Valid() {
		t.Fatal("generated CALPHAD terminal enums do not include the exact timeout tuple")
	}
}
