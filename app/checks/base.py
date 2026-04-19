from typing import Protocol
from pydantic import BaseModel


class Finding(BaseModel):
    passage: str
    issue: str
    suggested_fix: str
    rule_id: str
    position: int


class Check(Protocol):
    """Protocol for editorial checks."""

    async def run(self, text: str) -> list[Finding]:
        """Run the check and return findings."""
        ...
