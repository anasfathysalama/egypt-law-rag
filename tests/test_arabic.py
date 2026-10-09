from egylaw_rag.corpus.arabic import normalize_arabic, repair_line


def test_repair_line_restores_article_label_and_digits() -> None:
    assert repair_line("(١٠ ( ةدام") == "مادة (١٠)"


def test_repair_line_restores_article_one_opening() -> None:
    visual = "هذه اهلوانتت يتلا لئاسملا عيمج ىلع ةيعيرشتلا صوصنلا ىرست (١)"
    repaired = repair_line(visual)
    assert repaired.startswith("(١) ")
    assert "النصوص التشريعية" in repaired
    assert "المسائل" in repaired


def test_repair_line_fixes_lam_alef_without_touching_the_article() -> None:
    broken_relations = "\u0627\u0644\u0639\u0627\u0644\u0642\u0627\u062a"
    broken_islamic = "\u0627\u0644\u0625\u0633\u0627\u0644\u0645\u064a\u0629"
    assert repair_line(broken_relations[::-1]) == "العلاقات"
    assert repair_line(broken_islamic[::-1]) == "الإسلامية"
    assert repair_line("بالقانون"[::-1]) == "بالقانون"
    assert repair_line("القانون"[::-1]) == "القانون"


def test_normalize_arabic_unifies_alef_and_keeps_teh_marbuta() -> None:
    normalized = normalize_arabic("أَحْكَامُ إلى الشريعة")
    assert normalized == "احكام الى الشريعة"
    assert "ة" in normalized
    assert "ى" in normalized
