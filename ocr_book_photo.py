"""Command-line entry point for book/photo OCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytesseract

try:
    from .ocr_engine import run_tesseract
    from .postprocessing import apply_korean_spacing_model, build_paragraphs
    from .preprocessing import load_image, preprocess_for_tesseract, save_image
except ImportError:  # Allows `python log/ocr_book_photo.py ...`.
    from ocr_engine import run_tesseract
    from postprocessing import apply_korean_spacing_model, build_paragraphs
    from preprocessing import load_image, preprocess_for_tesseract, save_image


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
) -> None:
    original = load_image(input_path)
    # processed = preprocess_for_tesseract(original, crop_document=crop_document, crop_mode=crop_mode)
    processed = preprocess_for_tesseract(
    original,
    crop_document=crop_document,
    crop_mode=crop_mode,
    save_stages=True,
    stage_output_dir=Path("debug_stages"),
)
    save_image(processed_path, processed)

    tokens = run_tesseract(processed, lang=lang, psm=psm, oem=oem, min_confidence=min_confidence)
    text = build_paragraphs(tokens)
    if use_korean_spacing:
        text = apply_korean_spacing_model(text)

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
    )


if __name__ == "__main__":
    main()
