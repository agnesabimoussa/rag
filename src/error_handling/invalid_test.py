
class InvalidTest(Exception):
    def __init__(self,
                 message: str = "InvalidTest exception occured.") -> None:
        self.message = message
        super().__init__(self.message)
