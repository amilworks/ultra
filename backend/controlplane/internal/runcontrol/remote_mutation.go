package runcontrol

import "errors"

var (
	ErrInvalidRemoteMutationIntent = errors.New("invalid remote mutation intent")
	ErrEvaluationProfileMutation   = errors.New("protected evaluation profile forbids remote mutation intent")
)
