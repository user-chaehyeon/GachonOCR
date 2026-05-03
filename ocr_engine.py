"""Tesseract OCR adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass(frozen=True)
class OcrToken:
    text: str
    block_num: int
    par_num: int
    line_num: int
    left: int
    top: int
    width: int
    height: int
    confidence: float


def parse_confidence(value: str | int | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def run_tesseract(image: np.ndarray, lang: str, psm: int, oem: int, min_confidence: float) -> list[OcrToken]:
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)

    tokens: list[OcrToken] = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = parse_confidence(data["conf"][index])
        if not text or confidence < min_confidence:
            continue
        tokens.append(
            OcrToken(
                text=text,
                block_num=int(data["block_num"][index]),
                par_num=int(data["par_num"][index]),
                line_num=int(data["line_num"][index]),
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                confidence=confidence,
            )
        )
    return tokens
