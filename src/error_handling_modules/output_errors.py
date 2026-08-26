class OutputError(Exception):
    """Base class for errors raised while producing pipeline output."""

    def __init__(self,
                 message: str = "OutputError exception occured.") -> None:
        """Initialize the output error.

        Args:
            message: Human-readable explanation for the error.
        """
        self.message = message
        super().__init__(self.message)


class ApiCallError(OutputError):
    """Raised when a required external API credential/call is unavailable."""

    def __init__(self,
                 message: str = "ApiCallError exception occured.") -> None:
        """Initialize the API call error.

        Args:
            message: Human-readable explanation for the missing or failed API call.
        """
        super().__init__(message)
