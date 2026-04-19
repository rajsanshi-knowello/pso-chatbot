import re
from app.checks.base import Finding


class SpellingChecker:
    """Category 6: Spelling and Terminology checks."""

    # American spellings to flag (British/Australian preferred shown in comment)
    AMERICAN_SPELLINGS = {
        "organize": "organise",
        "color": "colour",
        "center": "centre",
        "analyze": "analyse",
        "behavior": "behaviour",
        "recognize": "recognise",
        "realize": "realise",
    }

    # Preferred PSO terminology
    TERMINOLOGY_RULES = {
        "programme": ("program", "PSO uses 'program'"),
        "portal": ("website", "Use 'website' instead of 'portal'"),
        "site": ("website", "Use 'website' instead of 'site'"),
        "e.g.": ("for example", "Use 'for example' in formal text"),
        "i.e.": ("that is", "Use 'that is' in formal text"),
        "commonwealth": ("Australian Government", "Use 'Australian Government'"),
        "federal": ("Australian Government", "Use 'Australian Government' where applicable"),
        "pre enrolment": ("pre-enrolment", "Use hyphenated form"),
        "preenrolment": ("pre-enrolment", "Use hyphenated form"),
    }

    async def run(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        findings.extend(await self._check_american_spellings(text))
        findings.extend(await self._check_terminology(text))

        return findings

    async def _check_american_spellings(self, text: str) -> list[Finding]:
        """Flag American spellings that should be Australian English."""
        findings: list[Finding] = []

        for american, australian in self.AMERICAN_SPELLINGS.items():
            pattern = r"\b" + re.escape(american) + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                context_start = max(0, pos - 20)
                context_end = min(len(text), pos + len(american) + 20)
                passage = text[context_start:context_end]

                findings.append(
                    Finding(
                        passage=passage,
                        issue=f"American spelling '{american}'; Australian English uses '{australian}'",
                        suggested_fix=f"Change '{american}' to '{australian}'",
                        rule_id="6.1-american-spellings",
                        position=pos,
                    )
                )

        return findings

    async def _check_terminology(self, text: str) -> list[Finding]:
        """Flag non-preferred PSO terminology."""
        findings: list[Finding] = []

        for term, (preferred, note) in self.TERMINOLOGY_RULES.items():
            # Handle special cases like "e.g." and "i.e." with dots
            if term in ("e.g.", "i.e."):
                pattern = re.escape(term)
            else:
                pattern = r"\b" + re.escape(term) + r"\b"

            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                context_start = max(0, pos - 20)
                context_end = min(len(text), pos + len(term) + 20)
                passage = text[context_start:context_end]

                findings.append(
                    Finding(
                        passage=passage,
                        issue=f"Terminology: '{term}' should be '{preferred}' ({note})",
                        suggested_fix=f"Replace '{term}' with '{preferred}'",
                        rule_id="6.2-terminology",
                        position=pos,
                    )
                )

        return findings
