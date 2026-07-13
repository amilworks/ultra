"""Typed failures for the isolated kinetics contract."""


class KineticsError(Exception):
    """Base class for a user-safe kinetics failure."""

    code = "kinetics_error"


class KineticsInputError(KineticsError):
    """The closed request or provenance contract is invalid."""

    code = "invalid_request"


class KineticsUnsupportedError(KineticsError):
    """The requested physical model is outside the qualified envelope."""

    code = "unsupported_model"


class KineticsExecutionError(KineticsError):
    """The solver failed or returned invalid scientific evidence."""

    code = "solver_failure"


class KineticsTimeoutError(KineticsExecutionError):
    """The bounded solver wall time elapsed."""

    code = "wall_time_exceeded"
