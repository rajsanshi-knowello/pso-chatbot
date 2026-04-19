import re
from app.checks.base import Finding


class ToneChecker:
    """Category 1: Tone of Voice checks."""

    # Banned promotional words (case-insensitive)
    BANNED_WORDS = {
        "excited",
        "proud",
        "thrilled",
        "leading",
        "game-changing",
        "transforming",
        "cutting-edge",
    }

    async def run(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        # Build regex pattern for banned words (word boundary, case-insensitive)
        pattern = r"\b(" + "|".join(re.escape(word) for word in self.BANNED_WORDS) + r")\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            word = match.group(1)
            pos = match.start()

            # Extract ~40 char context
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(word) + 20)
            passage = text[context_start:context_end]

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Promotional language: '{word}' should be more neutral and evidence-based",
                    suggested_fix=f"Replace '{word}' with neutral language",
                    rule_id="1.1-promotional-words",
                    position=pos,
                )
            )

        return findings
