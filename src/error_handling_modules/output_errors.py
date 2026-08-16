class OutputError(Exception):
    """Base class for errors raised while producing pipeline output."""

    def __init__(self,
                 message: str = "OutputError exception occured.") -> None:
        self.message = message
        super().__init__(self.message)


class ApiCallError(OutputError):
    """Raised when a required external API credential/call is unavailable."""

    def __init__(self,
                 message: str = "ApiCallError exception occured.") -> None:
        super().__init__(message)
