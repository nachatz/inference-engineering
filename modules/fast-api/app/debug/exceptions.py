class ProcUnavailable(Exception):
    def __init__(self) -> None:
        super().__init__("this needs Linux: PSS comes from /proc/<pid>/smaps")
