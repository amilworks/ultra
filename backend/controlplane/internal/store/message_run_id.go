package store

import (
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func threadMessageRunID(message domain.ThreadMessage, currentRunID string) string {
	if message.RunID != "" {
		return message.RunID
	}
	if strings.EqualFold(message.Role, "user") {
		return currentRunID
	}
	return ""
}
