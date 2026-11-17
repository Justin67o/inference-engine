from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    WAITING, RUNNING, FINISHED = range(3)


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = False
    output_ids: list[int] = field(default_factory=list)
    state: State = State.WAITING
    cache: object = None    # a KVCache from the pool while RUNNING
    on_token: object = None # callback(token_id, done), called on the engine thread


    @property
    def length(self):       # prompt + output tokens so far
        return len(self.prompt_ids) + len(self.output_ids)

    