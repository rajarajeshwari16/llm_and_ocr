import logging
from pathlib import Path
from typing import Dict, List, Sequence

import fitz
from PIL import Image


LOGGER = logging.getLogger("pdf_translate.rebuild")


def image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def wrap_text_to_width(page: fitz.Page, text: str, fontname: str, fontsize: float, max_width: float) -> str:
    words = text.split()
    if not words:
        return ""

    lines: List[str] = []
    current_line = words[0]

    for word in words[1:]:
        candidate = f"{current_line} {word}"
        candidate_width = fitz.get_text_length(candidate, fontname=fontname, fontsize=fontsize)
        if candidate_width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return "\n".join(lines)


def approximate_font_size(bbox_height: float, scale_factor: float, min_size: float, max_size: float) -> float:
    estimated = bbox_height * scale_factor
    return max(min_size, min(max_size, estimated))


def text_fits_box(line_count: int, fontsize: float, box_height: float, line_spacing: float = 1.2) -> bool:
    required_height = max(1, line_count) * fontsize * line_spacing
    return required_height <= box_height


def rebuild_translated_pdf(
    page_image_paths: Sequence[Path],
    translated_segments: Sequence[Dict],
    output_pdf_path: Path,
    rebuild_config: Dict,
) -> None:
    doc = fitz.open()
    font_name = rebuild_config.get("font_name", "helv")
    font_size_scale = rebuild_config.get("font_size_scale", 0.75)
    min_font_size = rebuild_config.get("min_font_size", 8)
    max_font_size = rebuild_config.get("max_font_size", 20)
    overlay_fill = tuple(rebuild_config.get("overlay_fill_color", [1, 1, 1]))
    text_color = tuple(rebuild_config.get("text_color", [0, 0, 0]))
    text_margin = rebuild_config.get("text_margin", 2)
    preserve_background = rebuild_config.get("preserve_background", False)

    segments_by_page: Dict[int, List[Dict]] = {}
    for segment in translated_segments:
        segments_by_page.setdefault(int(segment["page"]), []).append(segment)

    for page_number, image_path in enumerate(page_image_paths, start=1):
        width, height = image_dimensions(image_path)
        page = doc.new_page(width=width, height=height)
        page_segments = segments_by_page.get(page_number, [])

        # If no translated segments or all are empty, keep the original image as-is
        has_content = any((seg.get("translated_text") or "").strip() for seg in page_segments)
        if not has_content:
            page.insert_image(page.rect, filename=str(image_path))
            continue

        if preserve_background:
            page.insert_image(page.rect, filename=str(image_path))
        else:
            page.draw_rect(page.rect, color=overlay_fill, fill=overlay_fill, overlay=False)

        for segment in page_segments:
            translated_text = (segment.get("translated_text") or "").strip()
            if not translated_text:
                continue

            x, y, w, h = segment["bbox"]
            rect = fitz.Rect(x, y, x + w, y + h)
            text_rect = fitz.Rect(
                rect.x0 + text_margin,
                rect.y0 + text_margin,
                rect.x1 - text_margin,
                rect.y1 - text_margin,
            )
            font_size = approximate_font_size(h, font_size_scale, min_font_size, max_font_size)
            wrapped_text = wrap_text_to_width(page, translated_text, font_name, font_size, text_rect.width)
            while font_size > min_font_size:
                line_count = max(1, len(wrapped_text.splitlines()))
                if text_fits_box(line_count, font_size, text_rect.height):
                    break
                font_size -= 0.5
                wrapped_text = wrap_text_to_width(page, translated_text, font_name, font_size, text_rect.width)

            page.draw_rect(rect, color=overlay_fill, fill=overlay_fill, overlay=preserve_background)
            page.insert_textbox(
                text_rect,
                wrapped_text,
                fontsize=font_size,
                fontname=font_name,
                color=text_color,
                align=fitz.TEXT_ALIGN_LEFT,
                overlay=True,
            )

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf_path), deflate=True)
    doc.close()
    LOGGER.info("Rebuilt translated PDF saved to %s", output_pdf_path)
