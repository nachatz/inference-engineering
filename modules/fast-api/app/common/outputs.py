import re
from abc import abstractmethod

from pydantic import BaseModel
from pydantic import ConfigDict


class Output(BaseModel):
    """Something worth logging: `logger.info(output)` is all a caller writes.

    Its repr is the text log line. In JSON logs its fields become keys of the line, next to its `event`: the class
    name in snake case, so `WorkerReady` logs as `worker_ready`.
    """

    model_config = ConfigDict(frozen=True)

    @property
    def event(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()

    @abstractmethod
    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return repr(self)
