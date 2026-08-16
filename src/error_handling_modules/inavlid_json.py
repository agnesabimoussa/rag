class InvalidJSON(Exception):
    def __init__(self,
                 message="InvalidJSON exception occured.") -> None:
        self.message = message
        super().__init__(self.message)
