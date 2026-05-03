"""Paragraph reconstruction, OCR cleanup, and optional language-model correction."""

from __future__ import annotations

import re
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from .ocr_engine import OcrToken
except ImportError:  # Allows running modules directly from the log folder.
    from ocr_engine import OcrToken


HANGUL_RE = re.compile(r"[\u3130-\u318f\uac00-\ud7a3]")
TECH_TERM_RE = r"(?:PUSCH|PUCCH|PDCCH|DCI|SRI|SRS|ACK|NACK|NR|UL|DL|P_CMAX|P_TMAX)"


OCR_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"(?i)\byoga\s+전력\s+및\s+이앙\s+aol\s*[『「]?", "상향링크 전력 및 타이밍 제어"),
    (r"\b45\.1\.(\d+)", r"15.1.\1"),
    (r"22\s+CS\s+We\s+BE\s+BF\s+ga", "15.1.2.2 다중 개방 루프 변수"),
    (r"75723\s+OF\s+Dey\s+루프\s*프로스", "15.1.2.3 다중 폐쇄 루프 프로세스"),
    (r"(?:삼향링크|삼항링크|상링링크)", "상향링크"),
    (r"(?:하향림크|하향링 크)", "하향링크"),
    (r"(?:가방|기방|기발)\s*루프", "개방 루프"),
    (r"개방루프", "개방 루프"),
    (r"폐쇄루프", "폐쇄 루프"),
    (r"루프프로세스", "루프 프로세스"),
    (r"(?:랜덤|랩덤)\s*엑세스", "랜덤 액세스"),
    (r"스\s*케줄링", "스케줄링"),
    (r"그랜트없이", "그랜트 없이"),
    (r"경로\s*손실", "경로 손실"),
    (r"송신\s*전력", "송신 전력"),
    (r"전력\s*제어", "전력 제어"),
    (r"상향링크\s*스케줄링", "상향링크 스케줄링"),
    (r"하향링크\s*스케줄링", "하향링크 스케줄링"),
    (r"(?<![A-Za-z0-9])(?:PUSC\s*H|PUSCI[lI1]?|PUSC11|Puscil|05011|208011)(?![A-Za-z0-9])", "PUSCH"),
    (r"(?<![A-Za-z0-9])(?:PUCC\s*H|PUCCHS?|70001|206041)(?![A-Za-z0-9])", "PUCCH"),
    (r"(?i)(?<![A-Za-z0-9])(?:PDC\s*CH|PDCC\s*H|Pocc)(?=\s*안의)", "PDCCH"),
    (r"(?i)(?<![A-Za-z0-9])Der(?=\s*[01]-[01])", "DCI"),
    (r"(?i)\bhybrid\s*[- ]?\s*ARQ\s*(?:ack|ACK)\b", "hybrid-ARQ ACK"),
    (r"hybrid-ARQ ACK\s+전송", "hybrid-ARQ ACK을 전송"),
    (r"(?i)\backS(?=\s*전송)", "ACK을"),
    (r"차\s*이점", "차이점"),
    (r"정우", "경우"),
    (r"죄대", "최대"),
    (r"낮취", "낮춰"),
    (r"이리한", "이러한"),
    (r"단맡", "단말"),
    (r"(?:케리어|캐리이)", "캐리어"),
    (r"힐리스", "릴리즈"),
)


SPACING_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"(할|될|볼|쓸|둘|줄|얻을|있을)\s*수", r"\1 수"),
    (r"수\s*있", "수 있"),
    (r"수\s*없", "수 없"),
    (r"될\s*때", "될 때"),
    (r"할\s*때", "할 때"),
    (r"있는\s*경우", "있는 경우"),
    (r"없는\s*경우", "없는 경우"),
    (r"하는\s*경우", "하는 경우"),
    (r"되는\s*경우", "되는 경우"),
    (r"사용될\s*수", "사용될 수"),
    (r"설명할\s*수", "설명할 수"),
    (r"전송될\s*수", "전송될 수"),
    (r"있을\s*수", "있을 수"),
)


@lru_cache(maxsize=1)
def get_korean_spacing_model():
    """Return an optional Korean spacing model when pykospacing is installed."""
    try:
        from pykospacing import Spacing
    except Exception:
        return None
    return Spacing()


def normalize_common_ocr_errors(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace("ㆍ", ".").replace("·", ".")
    text = re.sub(r"[ \t]+", " ", text)

    for pattern, replacement in OCR_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(rf"({TECH_TERM_RE})(?=[\u3130-\u318f\uac00-\ud7a3])", r"\1 ", text)
    text = re.sub(rf"([\u3130-\u318f\uac00-\ud7a3])(?={TECH_TERM_RE}\b)", r"\1 ", text)
    text = re.sub(r"\b(PUSCH|PUCCH|PDCCH|DCI|SRI|SRS|ACK|NACK|NR|UL|DL)\b", lambda m: m.group(1).upper(), text)
    return text


def apply_rule_based_spacing(text: str) -> str:
    text = normalize_common_ocr_errors(text)
    for pattern, replacement in SPACING_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"([가-힣])(?=(?:상향링크|하향링크|PUSCH|PUCCH|PDCCH|DCI)\b)", r"\1 ", text)
    text = re.sub(r"(?<=[가-힣])\s+(?=[.,;:!?])", "", text)
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    text = re.sub(r"([([{<])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}>])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def apply_korean_spacing_model(text: str) -> str:
    """Apply ML-based Korean spacing if available; otherwise use rule-based cleanup."""
    spacing = get_korean_spacing_model()
    if spacing is None:
        paragraphs = [apply_rule_based_spacing(paragraph) for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]
        return "\n\n".join(paragraphs).strip() + "\n"

    corrected_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        merged = apply_rule_based_spacing(" ".join(lines))
        korean_chars = len(HANGUL_RE.findall(merged))
        if korean_chars < max(4, len(merged) * 0.20):
            corrected_paragraphs.append(merged)
            continue
        try:
            corrected_paragraphs.append(apply_rule_based_spacing(spacing(merged)))
        except Exception:
            corrected_paragraphs.append(merged)
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
    text = normalize_common_ocr_errors(text)
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


def is_heading_text(text: str) -> bool:
    if re.match(r"^\d{1,2}(?:\.\d+){2,4}\s+\S+", text):
        return True
    if len(text) <= 34 and re.search(r"(전력 제어|루프 변수|루프 프로세스|타이밍 제어)$", text):
        return True
    return False


def is_page_number_text(text: str, top: int, page_height: int) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", text.strip())) and top > page_height * 0.82


def is_sentence_end(text: str) -> bool:
    return bool(re.search(r"(다|요|임|함|됨|음|[.!?])['\")\]]?$", text.strip()))


def build_paragraphs(tokens: list[OcrToken]) -> str:
    if not tokens:
        return ""

    page_height = max(token.top + token.height for token in tokens)
    grouped: dict[tuple[int, int, int], list[OcrToken]] = {}
    for token in tokens:
        key = (token.block_num, token.par_num, token.line_num)
        grouped.setdefault(key, []).append(token)

    lines = []
    for key, line_tokens in grouped.items():
        top = min(token.top for token in line_tokens)
        left = min(token.left for token in line_tokens)
        right = max(token.left + token.width for token in line_tokens)
        height = max(token.height for token in line_tokens)
        text = line_text(line_tokens)
        if not text or is_page_number_text(text, top, page_height):
            continue
        lines.append({"key": key, "top": top, "left": left, "right": right, "height": height, "text": text})

    lines.sort(key=lambda item: (item["top"], item["left"]))
    if not lines:
        return ""

    median_height = float(np.median([line["height"] for line in lines]))
    median_left = float(np.median([line["left"] for line in lines]))

    paragraphs: list[list[str]] = []
    current: list[str] = []
    previous_line: dict[str, object] | None = None

    for line in lines:
        text = str(line["text"])
        heading = is_heading_text(text)
        new_paragraph = heading

        if previous_line is not None and not new_paragraph:
            previous_text = str(previous_line["text"])
            vertical_gap = int(line["top"]) - (int(previous_line["top"]) + int(previous_line["height"]))
            indent_delta = int(line["left"]) - int(previous_line["left"])
            different_tesseract_paragraph = line["key"][:2] != previous_line["key"][:2]  # type: ignore[index]
            first_line_indent = int(line["left"]) > median_left + median_height * 0.75

            if is_heading_text(previous_text):
                new_paragraph = True
            elif vertical_gap > median_height * 1.05:
                new_paragraph = True
            elif different_tesseract_paragraph and vertical_gap > median_height * 0.45:
                new_paragraph = True
            elif first_line_indent and is_sentence_end(previous_text) and vertical_gap > -median_height * 0.20:
                new_paragraph = True
            elif indent_delta > median_height * 1.8 and vertical_gap > median_height * 0.20:
                new_paragraph = True

        if new_paragraph and current:
            paragraphs.append(current)
            current = []

        current.append(text)
        previous_line = line

        if heading:
            paragraphs.append(current)
            current = []

    if current:
        paragraphs.append(current)

    paragraph_texts = [apply_rule_based_spacing(" ".join(paragraph)) for paragraph in paragraphs if paragraph]
    return "\n\n".join(paragraph_texts).strip() + "\n"


def restore_paragraph_breaks(text: str) -> str:
    text = re.sub(r"\s+(?=\d{1,2}(?:\.\d+){2,4}\s+\S+)", "\n\n", text)
    text = re.sub(r"\s+(?=[□■ㅁ]\s+)", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n\n".join(part.strip() for part in re.split(r"\n\s*\n", text) if part.strip())


def _slm_prompt(chunk: str) -> str:
    return (
        "다음 텍스트는 한국어 기술 서적을 OCR한 결과입니다. "
        "오타, OCR 오인식, 띄어쓰기, 문단만 교정하세요. "
        "원문에 없는 내용을 추가하거나 요약하지 마세요. "
        "PUSCH, PUCCH, PDCCH, DCI, SRI, NR, hybrid-ARQ ACK 같은 통신 용어는 보존하세요.\n\n"
        f"OCR:\n{chunk}\n\n교정문:\n"
    )


def _text_chunks(text: str, max_chars: int = 1800) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        elif current:
            current += "\n\n" + paragraph
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def apply_slm_correction(text: str, model: str | None = None, max_new_tokens: int = 900) -> str:
    """Optionally correct text with a local SLM.

    GGUF files use llama-cpp-python. Other model names/paths use transformers.
    The function is intentionally optional, so the OCR pipeline can run without SLM dependencies.
    """
    if not model:
        return text

    model_path = Path(model)
    corrected_chunks: list[str] = []
    if model_path.suffix.lower() == ".gguf":
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("Install llama-cpp-python to use a GGUF SLM model.") from exc

        llm = Llama(model_path=str(model_path), n_ctx=4096, verbose=False)
        for chunk in _text_chunks(text):
            response = llm(_slm_prompt(chunk), max_tokens=max_new_tokens, temperature=0.1, stop=["</s>"])
            corrected_chunks.append(response["choices"][0]["text"].strip())
        return "\n\n".join(corrected_chunks).strip() + "\n"

    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("Install transformers and a local Korean-capable model to use --slm-model.") from exc

    try:
        generator = pipeline("text2text-generation", model=model)
        for chunk in _text_chunks(text):
            result = generator(_slm_prompt(chunk), max_new_tokens=max_new_tokens, do_sample=False)[0]
            corrected_chunks.append(result["generated_text"].strip())
    except Exception:
        generator = pipeline("text-generation", model=model)
        for chunk in _text_chunks(text):
            prompt = _slm_prompt(chunk)
            result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
            generated = result["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt) :]
            corrected_chunks.append(generated.strip())

    return "\n\n".join(corrected_chunks).strip() + "\n"


def postprocess_ocr_text(
    text: str,
    use_korean_spacing: bool = True,
    slm_model: str | None = None,
) -> str:
    text = restore_paragraph_breaks(normalize_common_ocr_errors(text))
    if use_korean_spacing:
        text = apply_korean_spacing_model(text)
    else:
        text = apply_rule_based_spacing(text) + "\n"

    text = restore_paragraph_breaks(normalize_common_ocr_errors(text))
    if slm_model:
        try:
            text = apply_slm_correction(text, slm_model)
        except RuntimeError:
            raise
        except Exception as exc:
            warnings.warn(f"SLM correction failed; keeping rule-based OCR output: {exc}", RuntimeWarning)
    return text.strip() + "\n"
