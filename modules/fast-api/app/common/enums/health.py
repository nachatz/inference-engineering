from enum import StrEnum


class HealthStatus(StrEnum):
    OK = "ok"


class ReadyStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not ready"
