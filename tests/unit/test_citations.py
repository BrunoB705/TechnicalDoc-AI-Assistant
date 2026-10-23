from app.rag.citations import extract_citations, strip_invalid_citations


def test_extract_citations_variants():
    assert extract_citations("A [1][2]. B [3, 1]. C [2].") == [1, 2, 3]
    assert extract_citations("sin citas") == []
    assert extract_citations("lista [a] o [1.5]") == []


def test_strip_invalid_citations():
    assert strip_invalid_citations("Dato [1][9].", {1, 2}) == "Dato [1]."
    assert strip_invalid_citations("Dato [9].", {1}) == "Dato."
    assert strip_invalid_citations("Dato [1, 9].", {1}) == "Dato [1]."
