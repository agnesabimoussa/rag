class InvalidJSON(Exception):
    """Raised when a JSON file is missing, malformed, or fails schema
    validation against the expected pydantic model."""

    def __init__(self,
                 message: str = "InvalidJSON exception occured.") -> None:
        self.message = message
        super().__init__(self.message)
