import re
from app.checks.base import Finding


class AccessibilityChecker:
    """Category 2: Accessibility and Plain English checks."""

    async def run(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        # Check 1: Average sentence length
        findings.extend(await self._check_sentence_length(text))

        # Check 2: "click here" as link text
        findings.extend(await self._check_click_here(text))

        # Check 3: ALL CAPS paragraphs
        findings.extend(await self._check_all_caps(text))

        return findings

    async def _check_sentence_length(self, text: str) -> list[Finding]:
        """Flag sentences over 30 words and check average."""
        findings: list[Finding] = []

        # Split into sentences (simple heuristic)
        sentences = re.split(r"[.!?]+", text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]

        if not valid_sentences:
            return findings

        # Calculate average word count
        word_counts = [len(s.split()) for s in valid_sentences]
        avg_words = sum(word_counts) / len(word_counts)

        # Flag if average > 20
        if avg_words > 20:
            findings.append(
                Finding(
                    passage=text[:100],
                    issue=f"Average sentence length is {avg_words:.1f} words (target: 15-20)",
                    suggested_fix="Break up long sentences to improve readability",
                    rule_id="2.1-avg-sentence-length",
                    position=0,
                )
            )

        # Flag individual sentences over 30 words
        for sentence in valid_sentences:
            word_count = len(sentence.split())
            if word_count > 30:
                pos = text.find(sentence)
                findings.append(
                    Finding(
                        passage=sentence[:80],
                        issue=f"Sentence is {word_count} words (max: 30 recommended)",
                        suggested_fix="Split into shorter sentences",
                        rule_id="2.2-long-sentence",
                        position=pos,
                    )
                )

        return findings

    async def _check_click_here(self, text: str) -> list[Finding]:
        """Flag 'click here' as link text."""
        findings: list[Finding] = []

        # Look for variations of "click here"
        pattern = r"click\s+here"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + 30)
            passage = text[context_start:context_end]

            findings.append(
                Finding(
                    passage=passage,
                    issue="'Click here' is not descriptive link text",
                    suggested_fix="Use descriptive text that indicates where the link goes",
                    rule_id="2.3-click-here",
                    position=pos,
                )
            )

        return findings

    async def _check_all_caps(self, text: str) -> list[Finding]:
        """Flag ALL CAPS paragraphs (5+ consecutive uppercase words)."""
        findings: list[Finding] = []

        # Split into paragraphs
        paragraphs = text.split("\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if more than 5 consecutive uppercase words
            words = para.split()
            if len(words) < 5:
                continue

            # Count words that are ALL UPPERCASE (letters only, ignoring punctuation)
            uppercase_words = 0
            for w in words:
                # Remove punctuation and check if all letters are uppercase
                letters_only = ''.join(c for c in w if c.isalpha())
                if letters_only and letters_only.isupper():
                    uppercase_words += 1

            if uppercase_words >= 5:
                pos = text.find(para)
                findings.append(
                    Finding(
                        passage=para[:80],
                        issue="Paragraph uses ALL CAPS which affects readability",
                        suggested_fix="Convert to sentence case",
                        rule_id="2.4-all-caps",
                        position=pos,
                    )
                )

        return findings
