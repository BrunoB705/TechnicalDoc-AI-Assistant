import re

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str) -> list[int]:
    """Devuelve los índices citados ([1], [2][3], [1, 2]) en orden de aparición, sin duplicados."""
    seen: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for num in group.split(","):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


def strip_invalid_citations(text: str, valid: set[int]) -> str:
    """Elimina marcadores [n] que no corresponden a ninguna fuente entregada al modelo."""

    def repl(match: re.Match) -> str:
        nums = [int(n) for n in match.group(1).split(",")]
        kept = [n for n in nums if n in valid]
        return f"[{', '.join(map(str, kept))}]" if kept else ""

    cleaned = _CITATION_GROUP.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
