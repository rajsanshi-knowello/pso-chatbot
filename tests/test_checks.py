import pytest
from app.checks.tone import ToneChecker
from app.checks.accessibility import AccessibilityChecker
from app.checks.punctuation import PunctuationChecker
from app.checks.spelling import SpellingChecker
from app.checks.numbers import NumbersChecker
from app.checks.runner import CheckRunner


class TestToneChecker:
    @pytest.mark.anyio
    async def test_banned_words(self):
        checker = ToneChecker()
        text = "We are proud and thrilled to announce this game-changing initiative."
        findings = await checker.run(text)
        assert len(findings) == 3  # proud, thrilled, game-changing
        assert any("proud" in f.issue for f in findings)
        assert any("thrilled" in f.issue for f in findings)

    @pytest.mark.anyio
    async def test_no_banned_words(self):
        checker = ToneChecker()
        text = "We announce this new initiative to support workforce development."
        findings = await checker.run(text)
        assert len(findings) == 0


class TestAccessibilityChecker:
    @pytest.mark.anyio
    async def test_sentence_too_long(self):
        checker = AccessibilityChecker()
        text = "The organisation is working with stakeholders to develop and implement comprehensive strategies that support the achievement of goals and objectives across multiple regions and demographic areas."
        findings = await checker.run(text)
        # Should flag long sentences
        assert any("words" in f.issue for f in findings)

    @pytest.mark.anyio
    async def test_click_here(self):
        checker = AccessibilityChecker()
        text = "For more information, click here to view the policy document."
        findings = await checker.run(text)
        assert any("click here" in f.issue.lower() for f in findings)

    @pytest.mark.anyio
    async def test_all_caps(self):
        checker = AccessibilityChecker()
        text = "IMPORTANT NOTICE: ALL USERS MUST COMPLY WITH REQUIREMENTS."
        findings = await checker.run(text)
        assert any("ALL CAPS" in f.issue for f in findings)

    @pytest.mark.anyio
    async def test_compliant_text(self):
        checker = AccessibilityChecker()
        text = "This is a short, clear sentence. It meets guidelines well."
        findings = await checker.run(text)
        assert len(findings) == 0


class TestPunctuationChecker:
    @pytest.mark.anyio
    async def test_double_quotes(self):
        checker = PunctuationChecker()
        text = 'The report states: "This is important data."'
        findings = await checker.run(text)
        assert any("double" in f.issue.lower() for f in findings)

    @pytest.mark.anyio
    async def test_exclamation_marks(self):
        checker = PunctuationChecker()
        text = "This is urgent! We must act now!"
        findings = await checker.run(text)
        assert len([f for f in findings if "exclamation" in f.rule_id]) == 2

    @pytest.mark.anyio
    async def test_ampersands(self):
        checker = PunctuationChecker()
        text = "The policy covers education & training services."
        findings = await checker.run(text)
        assert any("&" in f.passage for f in findings)

    @pytest.mark.anyio
    async def test_negative_contractions(self):
        checker = PunctuationChecker()
        text = "We shouldn't and won't accept this proposal."
        findings = await checker.run(text)
        assert len([f for f in findings if "5.5" in f.rule_id]) >= 2

    @pytest.mark.anyio
    async def test_formal_contractions(self):
        checker = PunctuationChecker()
        text = "It's important that we're prepared. They're ready."
        findings = await checker.run(text)
        assert len([f for f in findings if "5.6" in f.rule_id]) >= 2


class TestSpellingChecker:
    @pytest.mark.anyio
    async def test_american_spellings(self):
        checker = SpellingChecker()
        text = "We must organize and analyze the data to recognize patterns."
        findings = await checker.run(text)
        assert len(findings) >= 3  # organize, analyze, recognize

    @pytest.mark.anyio
    async def test_terminology(self):
        checker = SpellingChecker()
        text = "Visit our portal and website. The e.g. example shows this."
        findings = await checker.run(text)
        assert any("portal" in f.issue.lower() for f in findings)
        assert any("e.g." in f.issue for f in findings)

    @pytest.mark.anyio
    async def test_pre_enrolment(self):
        checker = SpellingChecker()
        text = "Pre enrolment and preenrolment are incorrect forms."
        findings = await checker.run(text)
        # Both should be flagged
        assert len(findings) >= 2


class TestNumbersChecker:
    @pytest.mark.anyio
    async def test_numerals_1_9(self):
        checker = NumbersChecker()
        text = "We have 3 objectives and 5 strategies."
        findings = await checker.run(text)
        assert len([f for f in findings if "7.1" in f.rule_id]) >= 2

    @pytest.mark.anyio
    async def test_sentence_starting_numeral(self):
        checker = NumbersChecker()
        text = "5 strategies are outlined below."
        findings = await checker.run(text)
        assert any("7.2" in f.rule_id for f in findings)

    @pytest.mark.anyio
    async def test_time_format(self):
        checker = NumbersChecker()
        text = "The meeting is at 12 pm tomorrow."
        findings = await checker.run(text)
        assert any("midday" in f.suggested_fix for f in findings)

    @pytest.mark.anyio
    async def test_percent_symbol(self):
        checker = NumbersChecker()
        text = "We achieved a 50% increase in participation."
        findings = await checker.run(text)
        assert any("per cent" in f.suggested_fix for f in findings)

    @pytest.mark.anyio
    async def test_date_format(self):
        checker = NumbersChecker()
        text = "On April 15, 2024, the policy was released."
        findings = await checker.run(text)
        assert any("14 April" in f.suggested_fix or "15 April" in f.suggested_fix for f in findings)


class TestCheckRunner:
    @pytest.mark.anyio
    async def test_runner_all_checks(self):
        runner = CheckRunner()
        text = "We're thrilled to announce this program! It's 5 key objectives are: 1) organize data, 2) analyze results. The site has a 30% improvement."
        findings_by_category = await runner.run_all(text)

        # Should have findings in multiple categories
        assert 1 in findings_by_category  # Tone (proud, thrilled, etc.)
        assert 5 in findings_by_category  # Punctuation (contractions)
        assert 6 in findings_by_category  # Spelling (organize)
        assert 7 in findings_by_category  # Numbers

    @pytest.mark.anyio
    async def test_compliant_document(self):
        runner = CheckRunner()
        text = "This policy outlines workforce development strategies. The program achieves measurable outcomes through evidence-based approaches."
        findings_by_category = await runner.run_all(text)

        # Should have minimal or no findings
        total_findings = sum(len(f) for f in findings_by_category.values())
        assert total_findings < 3  # Allow a couple minor issues


class TestIntegration:
    @pytest.mark.anyio
    async def test_with_docx_content(self):
        """Integration test using realistic document content."""
        runner = CheckRunner()
        text = """
Policy Document: Workforce Development

We are proud and thrilled to announce this game-changing program! The site contains
e.g. examples and details about enrolment procedures. We've prepared 5 key objectives:
1) Organize training data, 2) Analyze participation rates at 50%, 3) Recognize achievements.

Visit our portal for details. The meeting is at 12 pm on April 15, 2024.
Don't worry, it's important that we're ready.
        """
        findings_by_category = await runner.run_all(text)

        # Verify findings across categories
        tone_findings = findings_by_category.get(1, [])
        assert len(tone_findings) > 0  # Should find promotional words

        numbers_findings = findings_by_category.get(7, [])
        assert len(numbers_findings) > 0  # Should find time/numeral issues

        spelling_findings = findings_by_category.get(6, [])
        assert len(spelling_findings) > 0  # Should find terminology issues
