import re
from app.checks.base import Finding


class PunctuationChecker:
    """Category 5: Punctuation checks."""

    async def run(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        findings.extend(await self._check_double_quotes(text))
        findings.extend(await self._check_exclamation_marks(text))
        findings.extend(await self._check_ampersands(text))
        findings.extend(await self._check_em_dashes(text))
        findings.extend(await self._check_negative_contractions(text))
        findings.extend(await self._check_formal_contractions(text))

        return findings

    async def _check_double_quotes(self, text: str) -> list[Finding]:
        """Flag double quotes (should use single quotes in Australian English)."""
        findings: list[Finding] = []

        # Look for double-quoted strings
        pattern = r'"([^"]*)"'
        for match in re.finditer(pattern, text):
            pos = match.start()
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + match.end() - pos + 20)
            passage = text[context_start:context_end]
            content = match.group(1)

            findings.append(
                Finding(
                    passage=passage,
                    issue="Double quotes used; Australian English uses single quotes",
                    suggested_fix=f"Change to '{content}'",
                    rule_id="5.1-double-quotes",
                    position=pos,
                )
            )

        return findings

    async def _check_exclamation_marks(self, text: str) -> list[Finding]:
        """Flag exclamation marks in formal body text."""
        findings: list[Finding] = []

        # Find exclamation marks (excluding headings and titles)
        for match in re.finditer(r"!", text):
            pos = match.start()
            # Get surrounding context
            context_start = max(0, pos - 30)
            context_end = min(len(text), pos + 30)
            passage = text[context_start:context_end]

            findings.append(
                Finding(
                    passage=passage,
                    issue="Exclamation mark in formal body text",
                    suggested_fix="Replace with period or rephrase to avoid exclamation",
                    rule_id="5.2-exclamation-marks",
                    position=pos,
                )
            )

        return findings

    async def _check_ampersands(self, text: str) -> list[Finding]:
        """Flag ampersands used in formal writing (use 'and')."""
        findings: list[Finding] = []

        # Find ampersands (but not in URLs or common abbreviations like &nbsp;)
        pattern = r"\s&\s|&\s[A-Z]|[a-z]&"
        for match in re.finditer(pattern, text):
            pos = match.start()
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + 30)
            passage = text[context_start:context_end]

            findings.append(
                Finding(
                    passage=passage,
                    issue="Ampersand (&) used in formal text",
                    suggested_fix="Replace '&' with 'and'",
                    rule_id="5.3-ampersands",
                    position=pos,
                )
            )

        return findings

    async def _check_em_dashes(self, text: str) -> list[Finding]:
        """Flag overuse of em dashes (more than 3 per 500 words)."""
        findings: list[Finding] = []

        # Count em dashes
        em_dash_count = text.count("—")
        word_count = len(text.split())

        if word_count > 0 and em_dash_count > (word_count / 500) * 3:
            findings.append(
                Finding(
                    passage=text[:100],
                    issue=f"Em dash overuse: {em_dash_count} dashes in {word_count} words (target: max 3 per 500 words)",
                    suggested_fix="Replace some em dashes with periods or commas",
                    rule_id="5.4-em-dash-overuse",
                    position=0,
                )
            )

        return findings

    async def _check_negative_contractions(self, text: str) -> list[Finding]:
        """Flag negative contractions (don't, won't, can't, etc.) in formal text."""
        findings: list[Finding] = []

        negative_contractions = {"don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't"}
        pattern = r"\b(" + "|".join(re.escape(c) for c in negative_contractions) + r")\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            contraction = match.group(1)
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(contraction) + 20)
            passage = text[context_start:context_end]

            # Map to non-contracted form
            expansion = {
                "don't": "do not",
                "won't": "will not",
                "can't": "cannot",
                "shouldn't": "should not",
                "wouldn't": "would not",
                "couldn't": "could not",
            }
            expanded = expansion.get(contraction.lower(), contraction)

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Negative contraction '{contraction}' in formal text",
                    suggested_fix=f"Use '{expanded}' instead",
                    rule_id="5.5-negative-contractions",
                    position=pos,
                )
            )

        return findings

    async def _check_formal_contractions(self, text: str) -> list[Finding]:
        """Flag informal contractions in formal documents (it's, we're, they're, you're)."""
        findings: list[Finding] = []

        formal_contractions = {"it's", "we're", "they're", "you're"}
        pattern = r"\b(" + "|".join(re.escape(c) for c in formal_contractions) + r")\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            contraction = match.group(1)
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(contraction) + 20)
            passage = text[context_start:context_end]

            # Map to expanded form
            expansion = {"it's": "it is", "we're": "we are", "they're": "they are", "you're": "you are"}
            expanded = expansion.get(contraction.lower(), contraction)

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Informal contraction '{contraction}' in formal text",
                    suggested_fix=f"Use '{expanded}' instead",
                    rule_id="5.6-formal-contractions",
                    position=pos,
                )
            )

        return findings
