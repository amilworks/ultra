package store

import "github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"

func isTerminalRunStatus(status domain.RunStatus) bool {
	return status == domain.RunStatusSucceeded ||
		status == domain.RunStatusFailed ||
		status == domain.RunStatusCanceled
}
