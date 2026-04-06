import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import pytesseract


LOGGER = logging.getLogger("pdf_translate.ocr")


@dataclass
class OCRSegment:
    text: str
    bbox: List[int]
    page: int

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "page": self.page,
        }


@dataclass
class OCRWord:
    text: str
    bbox: List[int]


def configure_tesseract(ocr_config: Dict) -> None:
    tesseract_cmd = ocr_config.get("tesseract_cmd")
    if tesseract_cmd:
        normalized = str(tesseract_cmd).strip()
        if normalized.startswith('r"') and normalized.endswith('"'):
            normalized = normalized[2:-1]
        elif normalized.startswith("r'") and normalized.endswith("'"):
            normalized = normalized[2:-1]
        elif normalized.startswith('"') and normalized.endswith('"'):
            normalized = normalized[1:-1]
        elif normalized.startswith("'") and normalized.endswith("'"):
            normalized = normalized[1:-1]

        pytesseract.pytesseract.tesseract_cmd = str(Path(normalized).expanduser().resolve())


def validate_tesseract(ocr_config: Dict, lang: str) -> None:
    configure_tesseract(ocr_config)

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        configured = ocr_config.get("tesseract_cmd")
        detail = f"Configured path: {configured}" if configured else "No explicit tesseract_cmd configured."
        raise RuntimeError(
            "Tesseract executable was not found. Install Tesseract OCR and either add it to PATH "
            "or set ocr.tesseract_cmd in config.yaml. "
            f"{detail}"
        ) from exc

    try:
        available_langs = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise RuntimeError("Tesseract is installed, but its language list could not be read.") from exc

    if lang not in available_langs:
        raise RuntimeError(
            f"Tesseract language pack '{lang}' is not installed. Available languages: {', '.join(sorted(available_langs))}"
        )


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_words(image, lang: str, tesseract_config: str) -> List[OCRWord]:
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=tesseract_config,
        output_type=pytesseract.Output.DICT,
    )

    words: List[OCRWord] = []
    total_items = len(data["text"])
    for i in range(total_items):
        text = (data["text"][i] or "").strip()
        confidence = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else -1
        width = int(data["width"][i])
        height = int(data["height"][i])
        if not text or confidence < 0 or width <= 0 or height <= 0:
            continue

        words.append(
            OCRWord(
                text=text,
                bbox=[
                    int(data["left"][i]),
                    int(data["top"][i]),
                    width,
                    height,
                ],
            )
        )

    return words


def should_join_word(previous_bbox: Sequence[int], current_bbox: Sequence[int], line_y_tolerance: float, gap_tolerance: float) -> bool:
    prev_x, prev_y, prev_w, prev_h = previous_bbox
    x, y, _, h = current_bbox
    prev_center_y = prev_y + (prev_h / 2.0)
    center_y = y + (h / 2.0)
    same_line = abs(prev_center_y - center_y) <= max(prev_h, h) * line_y_tolerance
    gap = x - (prev_x + prev_w)
    return same_line and gap <= max(prev_h, h) * gap_tolerance


def group_words_into_segments(
    words: Sequence[OCRWord],
    page: int,
    line_y_tolerance: float,
    gap_tolerance: float,
    paragraph_gap_multiplier: float,
) -> List[OCRSegment]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda item: (item.bbox[1], item.bbox[0]))
    lines: List[List[OCRWord]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        last_word = lines[-1][-1]
        if should_join_word(last_word.bbox, word.bbox, line_y_tolerance, gap_tolerance):
            lines[-1].append(word)
        else:
            lines.append([word])

    segments: List[OCRSegment] = []
    paragraph_words: List[OCRWord] = []
    previous_line_bottom = None
    previous_line_height = None

    for line_words in lines:
        line_top = min(word.bbox[1] for word in line_words)
        line_bottom = max(word.bbox[1] + word.bbox[3] for word in line_words)
        line_height = max(word.bbox[3] for word in line_words)

        if (
            previous_line_bottom is not None
            and previous_line_height is not None
            and (line_top - previous_line_bottom) > previous_line_height * paragraph_gap_multiplier
            and paragraph_words
        ):
            segments.append(build_segment(paragraph_words, page))
            paragraph_words = []

        paragraph_words.extend(line_words)
        previous_line_bottom = line_bottom
        previous_line_height = line_height

    if paragraph_words:
        segments.append(build_segment(paragraph_words, page))

    return [segment for segment in segments if segment.text.strip()]


def build_segment(words: Sequence[OCRWord], page: int) -> OCRSegment:
    x0 = min(word.bbox[0] for word in words)
    y0 = min(word.bbox[1] for word in words)
    x1 = max(word.bbox[0] + word.bbox[2] for word in words)
    y1 = max(word.bbox[1] + word.bbox[3] for word in words)
    text = " ".join(word.text for word in words)
    return OCRSegment(text=text, bbox=[x0, y0, x1 - x0, y1 - y0], page=page)


def ocr_page_image(task: Tuple[str, int, str, Dict]) -> List[OCRSegment]:
    image_path, page_number, lang, ocr_config = task
    LOGGER.debug("OCR on page %s from %s", page_number, image_path)
    configure_tesseract(ocr_config)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load page image for OCR: {image_path}")

    processed = preprocess_image(image)
    words = extract_words(
        processed,
        lang=lang,
        tesseract_config=ocr_config.get("tesseract_config", "--oem 3 --psm 6"),
    )
    return group_words_into_segments(
        words,
        page=page_number,
        line_y_tolerance=ocr_config.get("line_y_tolerance", 0.6),
        gap_tolerance=ocr_config.get("word_gap_tolerance", 1.8),
        paragraph_gap_multiplier=ocr_config.get("paragraph_gap_multiplier", 1.5),
    )
