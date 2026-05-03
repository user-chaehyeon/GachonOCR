"""Paragraph reconstruction and Korean spacing correction."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

try:
    from .ocr_engine import OcrToken
except ImportError:  # Allows running modules directly from the log folder.
    from ocr_engine import OcrToken


@lru_cache(maxsize=1)
def get_korean_spacing_model():
    """Return an optional Korean spacing model when pykospacing is installed."""
    try:
        from pykospacing import Spacing
    except Exception:
        return None
    return Spacing()


def apply_korean_spacing_model(text: str) -> str:
    """Apply ML-based Korean spacing if available; otherwise keep rule-based output."""
    spacing = get_korean_spacing_model()
    if spacing is None:
        return text

    corrected_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        merged = " ".join(lines)
        # Keep very formula/code-heavy lines away from the Korean spacing model.
        korean_chars = len(re.findall(r"[?-??-??-?]", merged))
        if korean_chars < max(4, len(merged) * 0.20):
            corrected_paragraphs.append("\n".join(lines))
            continue
        try:
            corrected_paragraphs.append(spacing(merged))
        except Exception:
            corrected_paragraphs.append("\n".join(lines))
    return "\n\n".join(corrected_paragraphs).strip() + "\n"


def is_cjk_or_hangul(text: str) -> bool:
    return bool(re.search(r"[\u3130-\u318f\uac00-\ud7a3\u4e00-\u9fff]", text))


def is_latin_or_number(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))


def should_insert_space(previous: str, current: str, gap: int, median_char_width: float) -> bool:
    if not previous:
        return False

    no_space_before = set(".,;:!?%)]}>")
    no_space_after = set("([{<")
    math_symbols = set("=+-*/^<>")

    if current[:1] in no_space_before or previous[-1:] in no_space_after:
        return False

    if previous[-1:] in math_symbols or current[:1] in math_symbols:
        return True

    if re.search(r"[A-Za-z0-9]$", previous) and re.search(r"^[A-Za-z0-9]", current):
        return True

    previous_is_cjk = is_cjk_or_hangul(previous[-1:])
    current_is_cjk = is_cjk_or_hangul(current[:1])
    previous_is_alnum = is_latin_or_number(previous[-1:])
    current_is_alnum = is_latin_or_number(current[:1])

    if (previous_is_alnum and current_is_cjk) or (previous_is_cjk and current_is_alnum):
        return gap > max(3, median_char_width * 0.30)

    if previous_is_cjk and current_is_cjk:
        return gap > max(5, median_char_width * 0.52)

    return gap > max(4, median_char_width * 0.45)


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    text = re.sub(r"([([{<])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}>])", r"\1", text)
    text = re.sub(r"\s*([=+\-*/^<>])\s*", r" \1 ", text)
    text = re.sub(r"(?<=[0-9])\s*-\s*(?=[0-9])", "-", text)
    text = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", text)
    text = re.sub(r"([A-Za-z0-9])(?=[\u3130-\u318f\uac00-\ud7a3])", r"\1 ", text)
    text = re.sub(r"([\u3130-\u318f\uac00-\ud7a3])(?=[A-Za-z0-9])", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def line_text(tokens: list[OcrToken]) -> str:
    tokens = sorted(tokens, key=lambda token: token.left)
    if not tokens:
        return ""

    char_widths = [token.width / max(1, len(token.text)) for token in tokens]
    median_char_width = float(np.median(char_widths)) if char_widths else 8.0

    parts: list[str] = []
    previous: OcrToken | None = None
    for token in tokens:
        if previous is not None:
            gap = token.left - (previous.left + previous.width)
            if should_insert_space(previous.text, token.text, gap, median_char_width):
                parts.append(" ")
        parts.append(token.text)
        previous = token

    return normalize_text("".join(parts))


def build_paragraphs(tokens: list[OcrToken]) -> str:
    if not tokens:
        return ""

    grouped: dict[tuple[int, int, int], list[OcrToken]] = {}
    for token in tokens:
        key = (token.block_num, token.par_num, token.line_num)
        grouped.setdefault(key, []).append(token)

    lines = []
    for key, line_tokens in grouped.items():
        top = min(token.top for token in line_tokens)
        left = min(token.left for token in line_tokens)
        height = max(token.height for token in line_tokens)
        text = line_text(line_tokens)
        if text:
            lines.append({"key": key, "top": top, "left": left, "height": height, "text": text})

    lines.sort(key=lambda item: (item["top"], item["left"]))
    median_height = float(np.median([line["height"] for line in lines])) if lines else 20.0

    paragraphs: list[list[str]] = []
    current: list[str] = []
    previous_line: dict[str, object] | None = None

    for line in lines:
        new_paragraph = False
        if previous_line is not None:
            vertical_gap = int(line["top"]) - (int(previous_line["top"]) + int(previous_line["height"]))
            indent_delta = int(line["left"]) - int(previous_line["left"])
            different_tesseract_paragraph = line["key"][:2] != previous_line["key"][:2]  # type: ignore[index]

            # if different_tesseract_paragraph:
            #     new_paragraph = True
            # elif vertical_gap > median_height * 0.9:
            #     new_paragraph = True
            # elif indent_delta > median_height * 1.2:
            #     new_paragraph = True
            
            if vertical_gap > median_height * 1.4:
                new_paragraph = True
            elif indent_delta > median_height * 1.8 and vertical_gap > median_height * 0.4:
                new_paragraph = True
            elif different_tesseract_paragraph and vertical_gap > median_height * 0.9:
                new_paragraph = True

        if new_paragraph and current:
            paragraphs.append(current)
            current = []

        current.append(str(line["text"]))
        previous_line = line

    if current:
        paragraphs.append(current)

    # paragraph_texts = ["\n".join(paragraph) for paragraph in paragraphs]
    # return "\n\n".join(paragraph_texts).strip() + "\n"
    
    # paragraph_texts = [" ".join(paragraph) for paragraph in paragraphs]
    # return "\n\n".join(paragraph_texts).strip() + "\n"
    
    paragraph_texts = [" ".join(paragraph) for paragraph in paragraphs]
    new_sent = "\n\n".join(paragraph_texts).strip() + "\n"

    print(new_sent)

    try:
        from pykospacing import Spacing
        spacing = Spacing()
        kospacing_sent = spacing(new_sent)
        print(kospacing_sent)
        return kospacing_sent
    except Exception:
        return new_sent
