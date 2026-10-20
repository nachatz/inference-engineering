from enum import StrEnum


class Allocator(StrEnum):
    JEMALLOC = "jemalloc"
    TCMALLOC = "tcmalloc"
    MIMALLOC = "mimalloc"
    GLIBC = "glibc"
    UNKNOWN = "unknown"
