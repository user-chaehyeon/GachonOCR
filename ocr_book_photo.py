"""Command-line entry point for book/photo OCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pytesseract

try:
    from .ocr_engine import run_tesseract
    from .postprocessing import build_paragraphs, postprocess_ocr_text
    from .preprocessing import load_image, preprocess_pages_for_tesseract, save_image
except ImportError:  # Allows `python log/ocr_book_photo.py ...`.
    from ocr_engine import run_tesseract
    from postprocessing import build_paragraphs, postprocess_ocr_text
    from preprocessing import load_image, preprocess_pages_for_tesseract, save_image


def stack_pages_vertically(pages: list[np.ndarray]) -> np.ndarray:
    max_width = max(page.shape[1] for page in pages)
    spacer_height = max(24, max(page.shape[0] for page in pages) // 80)
    normalized_pages = []
    for page in pages:
        if page.shape[1] == max_width:
            normalized_pages.append(page)
            continue
        canvas = np.full((page.shape[0], max_width), 255, dtype=page.dtype)
        x = (max_width - page.shape[1]) // 2
        canvas[:, x : x + page.shape[1]] = page
        normalized_pages.append(canvas)

    spacer = np.full((spacer_height, max_width), 255, dtype=normalized_pages[0].dtype)
    output = normalized_pages[0]
    for page in normalized_pages[1:]:
        output = cv2.vconcat([output, spacer, page])
    return output


def save_processed_pages(processed_path: Path, pages: list[np.ndarray]) -> None:
    if len(pages) == 1:
        save_image(processed_path, pages[0])
        return

    suffix = processed_path.suffix or ".png"
    for index, page in enumerate(pages, start=1):
        page_path = processed_path.with_name(f"{processed_path.stem}_page_{index:02d}{suffix}")
        save_image(page_path, page)

    save_image(processed_path, stack_pages_vertically(pages))


def ocr_image(
    input_path: Path,
    output_path: Path,
    processed_path: Path,
    lang: str,
    psm: int,
    oem: int,
    min_confidence: float,
    crop_document: bool,
    crop_mode: str,
    use_korean_spacing: bool,
    slm_model: str | None,
) -> None:
    original = load_image(input_path)
    processed_pages = preprocess_pages_for_tesseract(
        original,
        crop_document=crop_document,
        crop_mode=crop_mode,
        split_pages=True,
        save_stages=True,
        stage_output_dir=Path("debug_stages"),
    )
    save_processed_pages(processed_path, processed_pages)

    page_texts = []
    for processed in processed_pages:
        tokens = run_tesseract(processed, lang=lang, psm=psm, oem=oem, min_confidence=min_confidence)
        text = build_paragraphs(tokens)
        if text:
            page_texts.append(text)

    text = postprocess_ocr_text(
        "\n\n".join(page_texts),
        use_korean_spacing=use_korean_spacing,
        slm_model=slm_model,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Korean/English text and formulas from photographed pages.")
    parser.add_argument("input", type=Path, help="Input image path.")
    parser.add_argument("--output", "-o", type=Path, default=Path("ocr_result.txt"), help="Output UTF-8 txt path.")
    parser.add_argument(
        "--processed",
        "-p",
        type=Path,
        default=Path("ocr_preprocessed.png"),
        help="Path for the image that will be passed to Tesseract.",
    )
    parser.add_argument("--lang", default="kor+eng", help="Tesseract language code. Default: kor+eng")
    parser.add_argument("--psm", type=int, default=4, help="Tesseract page segmentation mode. Default: 4")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode. Default: 3")
    parser.add_argument("--min-confidence", type=float, default=25.0, help="Drop OCR tokens below this confidence.")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic document/book perspective crop.")
    parser.add_argument(
        "--crop-mode",
        choices=("safe", "perspective"),
        default="safe",
        help="Use safe page crop by default; perspective applies four-point warping when detection is reliable.",
    )
    parser.add_argument(
        "--no-korean-spacing",
        action="store_true",
        help="Disable optional Korean spacing model postprocessing.",
    )
    parser.add_argument(
        "--slm-model",
        help="Optional local SLM model name/path for final OCR text correction. GGUF paths use llama-cpp-python.",
    )
    parser.add_argument("--tesseract-cmd", help="Full path to tesseract executable, useful on Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    ocr_image(
        input_path=args.input,
        output_path=args.output,
        processed_path=args.processed,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
        crop_document=not args.no_crop,
        crop_mode=args.crop_mode,
        use_korean_spacing=not args.no_korean_spacing,
        slm_model=args.slm_model,
    )


if __name__ == "__main__":
    main()
