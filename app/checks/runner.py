import asyncio
from app.checks.base import Finding
from app.checks.tone import ToneChecker
from app.checks.accessibility import AccessibilityChecker
from app.checks.punctuation import PunctuationChecker
from app.checks.spelling import SpellingChecker
from app.checks.numbers import NumbersChecker


class CheckRunner:
    """Orchestrates all editorial checks and returns findings by category."""

    def __init__(self):
        self.checkers = {
            1: ToneChecker(),
            2: AccessibilityChecker(),
            5: PunctuationChecker(),
            6: SpellingChecker(),
            7: NumbersChecker(),
        }

    async def run_all(self, text: str) -> dict[int, list[Finding]]:
        """Run all checks concurrently and return findings grouped by category."""
        tasks = {
            category: self.checkers[category].run(text) for category in self.checkers
        }

        results = await asyncio.gather(*tasks.values())

        findings_by_category: dict[int, list[Finding]] = {
            category: [] for category in self.checkers
        }

        for category, findings in zip(self.checkers.keys(), results):
            findings_by_category[category] = findings

        return findings_by_category
