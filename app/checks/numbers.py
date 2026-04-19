import re
from app.checks.base import Finding


class NumbersChecker:
    """Category 7: Numbers, Dates and Time checks."""

    async def run(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        findings.extend(await self._check_numerals_1_9(text))
        findings.extend(await self._check_sentence_starting_numeral(text))
        findings.extend(await self._check_time_format(text))
        findings.extend(await self._check_percent_symbol(text))
        findings.extend(await self._check_date_format(text))

        return findings

    async def _check_numerals_1_9(self, text: str) -> list[Finding]:
        """Flag numerals 1-9 used in body text (should be spelled out)."""
        findings: list[Finding] = []

        # Look for standalone digits 1-9 (but not as part of larger numbers)
        pattern = r"\b([1-9])\b"
        for match in re.finditer(pattern, text):
            pos = match.start()
            digit = match.group(1)

            # Get context
            context_start = max(0, pos - 30)
            context_end = min(len(text), pos + 30)
            passage = text[context_start:context_end]

            # Map to word form
            digit_words = {
                "1": "one",
                "2": "two",
                "3": "three",
                "4": "four",
                "5": "five",
                "6": "six",
                "7": "seven",
                "8": "eight",
                "9": "nine",
            }

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Numeral '{digit}' should be spelled out in formal body text",
                    suggested_fix=f"Change '{digit}' to '{digit_words[digit]}'",
                    rule_id="7.1-numerals-1-9",
                    position=pos,
                )
            )

        return findings

    async def _check_sentence_starting_numeral(self, text: str) -> list[Finding]:
        """Flag sentences starting with a numeral."""
        findings: list[Finding] = []

        # Find sentence boundaries and check if they start with digits
        sentences = re.split(r"([.!?])\s+", text)
        pos = 0

        for i, sentence in enumerate(sentences):
            if i % 2 == 1:  # Skip delimiters
                continue

            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if starts with digit
            if re.match(r"^\d", sentence):
                context_end = min(len(text), pos + len(sentence) + 20)
                passage = text[pos : context_end]

                # Get the leading digit(s)
                match = re.match(r"^(\d+)", sentence)
                numeral = match.group(1) if match else "?"

                findings.append(
                    Finding(
                        passage=passage,
                        issue=f"Sentence starts with numeral '{numeral}'",
                        suggested_fix=f"Spell out the number or rephrase sentence",
                        rule_id="7.2-sentence-starting-numeral",
                        position=pos,
                    )
                )

            pos += len(sentence) + 2

        return findings

    async def _check_time_format(self, text: str) -> list[Finding]:
        """Flag incorrect time formats (12 pm/am should be midday/midnight)."""
        findings: list[Finding] = []

        # Check for "12 pm", "12 am", "12pm", "12am"
        for match in re.finditer(r"12\s*(pm|am)", text, re.IGNORECASE):
            pos = match.start()
            time_str = match.group(0)
            period = match.group(1).lower()

            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(time_str) + 20)
            passage = text[context_start:context_end]

            suggested = "midday" if period == "pm" else "midnight"

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Time format '{time_str}' should be '{suggested}'",
                    suggested_fix=f"Replace '{time_str}' with '{suggested}'",
                    rule_id="7.3-time-format",
                    position=pos,
                )
            )

        return findings

    async def _check_percent_symbol(self, text: str) -> list[Finding]:
        """Flag % symbol in body text (should use 'per cent')."""
        findings: list[Finding] = []

        # Look for % not in tables/captions (simple heuristic: not in "Table" context)
        for match in re.finditer(r"(\d+\s*%)", text):
            pos = match.start()
            percent_str = match.group(1)

            # Skip if it looks like it's in a table context
            line_start = text.rfind("\n", 0, pos) + 1
            line = text[line_start : pos + 50]
            if "table" in line.lower() or "|" in line:
                continue

            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(percent_str) + 20)
            passage = text[context_start:context_end]

            # Extract just the number
            number = re.match(r"(\d+)", percent_str).group(1)

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Symbol '%' used in body text",
                    suggested_fix=f"Use 'per cent': '{number} per cent'",
                    rule_id="7.4-percent-symbol",
                    position=pos,
                )
            )

        return findings

    async def _check_date_format(self, text: str) -> list[Finding]:
        """Flag incorrect date formats (should be 'DD Month YYYY', not 'Month DD, YYYY')."""
        findings: list[Finding] = []

        # Look for "Month DD, YYYY" format
        month_pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        pattern = month_pattern + r"\s+(\d{1,2}),\s+(\d{4})"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            date_str = match.group(0)
            month = match.group(1)
            day = match.group(2)
            year = match.group(3)

            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(date_str) + 20)
            passage = text[context_start:context_end]

            correct_format = f"{day} {month} {year}"

            findings.append(
                Finding(
                    passage=passage,
                    issue=f"Date format '{date_str}' should be '{correct_format}'",
                    suggested_fix=f"Change to '{correct_format}' (no comma)",
                    rule_id="7.5-date-format",
                    position=pos,
                )
            )

        return findings
