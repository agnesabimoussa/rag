class InvalidJSON(Exception):
    """Raised when a JSON file is missing, malformed, or fails schema
    validation against the expected pydantic model."""

    def __init__(self,
                 message: str = "InvalidJSON exception occured.") -> None:
        """Initialize the invalid-JSON error.

        Args:
            message: Human-readable description of the validation failure.
        """
        self.message = message
        super().__init__(self.message)
