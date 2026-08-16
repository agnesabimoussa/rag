class OutputError(Exception):
    def __init__(self,
                 message="OutputError exception occured.") -> None:
        self.message = message
        super().__init__(self.message)

class ApiCallError(OutputError):
    def __init__(self,
                 message="ApiCallError exception occured.") -> None:
        super().__init__(message)
