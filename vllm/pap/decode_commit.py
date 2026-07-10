import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PAPDecodeCommit:
    request_id: str
    commit_seq: int
    new_seq_len: int
    new_token_ids: tuple[int, ...]
    layer_complete: bool

    def __post_init__(self) -> None:
        if self.commit_seq <= 0:
            raise ValueError("commit_seq must be positive")
        # Normalize new_token_ids to tuple so list users also get equality.
        if not isinstance(self.new_token_ids, tuple):
            object.__setattr__(self, "new_token_ids", tuple(self.new_token_ids))

    @classmethod
    def from_dict(cls, d: dict) -> "PAPDecodeCommit":
        return cls(
            request_id=str(d["request_id"]),
            commit_seq=int(d["commit_seq"]),
            new_seq_len=int(d["new_seq_len"]),
            new_token_ids=tuple(int(t) for t in d["new_token_ids"]),
            layer_complete=bool(d["layer_complete"]),
        )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "commit_seq": self.commit_seq,
            "new_seq_len": self.new_seq_len,
            "new_token_ids": list(self.new_token_ids),
            "layer_complete": self.layer_complete,
        }


def serialize_commit(commit: PAPDecodeCommit) -> bytes:
    return json.dumps(commit.to_dict()).encode("utf-8")


def deserialize_commit(blob: bytes) -> PAPDecodeCommit:
    return PAPDecodeCommit.from_dict(json.loads(blob.decode("utf-8")))
