from typing import Self

import psutil
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ProcessMemory(BaseModel):
    """One process's memory in bytes, as psutil reads it from /proc/<pid>/smaps. PSS needs Linux.

    `private` is psutil's USS (Private_Clean + Private_Dirty): the pages no other process maps, so what a worker
    costs on top of what it shares with the master.
    """

    model_config = ConfigDict(frozen=True, validate_by_name=True)

    pid: int
    rss: int
    pss: int
    private: int = Field(validation_alias="uss")

    @classmethod
    def of(cls, pid: int) -> Self:
        return cls.model_validate({"pid": pid, **psutil.Process(pid).memory_full_info()._asdict()})
