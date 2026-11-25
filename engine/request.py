from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    WAITING, RUNNING, FINISHED = range(3)


@dataclass(eq=False)
class Request:
    id: int
    prompt_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = False
    top_k: int = 0          # 0 = off
    top_p: float = 1.0      # 1.0 = off
    seed: int | None = None # None = use the shared generator
    stop: list[str] = field(default_factory=list)   # stop strings, matched on decoded output
    finish_reason: str | None = None    # "stop" or "length" once FINISHED
    output_ids: list[int] = field(default_factory=list)
    state: State = State.WAITING
    block_table: list[int] = field(default_factory=list)    # physical KV block ids, in order
    generator: object = None # a seeded torch.Generator, made by the engine if the seed is set
    on_token: object = None # callback(token_id, done), called on the engine thread
    draft_len: int = 0      # positions the draft model's cache holds; trails length after a verify
    spec_passes: int = 0    # verify passes that carried drafts
    spec_drafted: int = 0   # draft tokens sent to the target
    spec_accepted: int = 0  # drafts the target kept
    spec_rejected: int = 0  # passes that ended in a rejection (the rest accepted every draft)


    @property
    def length(self):       # prompt + output tokens so far
        return len(self.prompt_ids) + len(self.output_ids)

