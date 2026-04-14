import argparse
import logging
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml
from pdf2image import convert_from_path
from tqdm import tqdm

from ocr import OCRSegment, ocr_page_image, validate_tesseract
from rebuild import rebuild_translated_pdf
from translate import VertexTranslator, chunk_segments_for_translation


LOGGER = logging.getLogger("pdf_translate")
BASE_DIR = Path(__file__).resolve().parent

LANG_NAMES = {
    "kan": "Kannada",
    "hin": "Hindi",
    "tam": "Tamil",
    "tel": "Telugu",
    "eng": "English",
}


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate scanned Indian-language PDFs into English while keeping layout close to the original."
    )
    parser.add_argument("--input", required=True, help="Path to scanned PDF.")
    parser.add_argument("--output", required=True, help="Path for translated PDF.")
    parser.add_argument("--lang", required=True, choices=["hin", "kan", "tam", "tel", "eng"], help="Tesseract OCR language code.")
    parser.add_argument("--config", help="Path to YAML config file. Defaults to config.yaml next to main.py.")
    parser.add_argument("--project", help="Google Cloud project ID override.")
    parser.add_argument("--location", help="Vertex AI location override, e.g. us-central1.")
    parser.add_argument("--model", help="Vertex AI model override.")
    parser.add_argument("--dpi", type=int, help="PDF rendering DPI override.")
    parser.add_argument("--max-workers", type=int, help="Parallel OCR worker count override.")
    return parser.parse_args()


def resolve_config_path(config_arg: Optional[str]) -> Path:
    if config_arg:
        return Path(config_arg).expanduser().resolve()
    return BASE_DIR / "config.yaml"


def resolve_config_relative_path(base_dir: Path, raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    config_relative = (base_dir / candidate).resolve()
    if config_relative.exists():
        return str(config_relative)

    parent_relative = (base_dir.parent / candidate).resolve()
    if parent_relative.exists():
        return str(parent_relative)

    return str(config_relative)


def save_page_images(input_pdf: Path, image_dir: Path, dpi: int, fmt: str) -> List[Path]:
    LOGGER.info("Rendering PDF pages to images at %s DPI", dpi)
    pil_images = convert_from_path(
        str(input_pdf),
        dpi=dpi,
        fmt=fmt,
        poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin",
    )
    image_paths: List[Path] = []

    for page_index, image in enumerate(tqdm(pil_images, desc="Saving page images"), start=1):
        image_path = image_dir / f"page_{page_index:04d}.{fmt}"
        image.save(image_path)
        image_paths.append(image_path)

    return image_paths


def ocr_pages_in_parallel(
    image_paths: Sequence[Path],
    lang: str,
    config: Dict,
    max_workers: int,
) -> List[List[OCRSegment]]:
    LOGGER.info("Running OCR across %s pages with %s workers", len(image_paths), max_workers)
    tasks = [
        (
            str(image_path),
            index + 1,
            lang,
            config.get("ocr", {}),
        )
        for index, image_path in enumerate(image_paths)
    ]

    try:
        with mp.Pool(processes=max_workers) as pool:
            page_segments = list(
                tqdm(
                    pool.imap(ocr_page_image, tasks),
                    total=len(tasks),
                    desc="OCR pages",
                )
            )
        return page_segments
    except (PermissionError, OSError) as exc:
        LOGGER.warning(
            "Parallel OCR could not start (%s). Falling back to sequential OCR.",
            exc,
        )
        return [
            ocr_page_image(task)
            for task in tqdm(tasks, total=len(tasks), desc="OCR pages")
        ]


def flatten(nested_segments: Sequence[Sequence[OCRSegment]]) -> List[OCRSegment]:
    flattened: List[OCRSegment] = []
    for page_segments in nested_segments:
        flattened.extend(page_segments)
    return flattened


def vision_ocr_pages(
    image_paths: Sequence[Path],
    translator,
    lang_name: str,
    max_workers: int = 4,
    english_only: bool = False,
) -> List[Dict]:
    from PIL import Image as PILImage
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_page(args):
        page_number, image_path = args
        with PILImage.open(image_path) as img:
            img_w, img_h = img.size
        blocks = translator.ocr_translate_page_image(str(image_path), page_number, lang_name, english_only=english_only)
        non_empty = sum(1 for b in blocks if (b.get("translated_text") or "").strip())
        LOGGER.info("Page %s: %s blocks extracted, %s non-empty", page_number, len(blocks), non_empty)
        page_segments = []
        for block in blocks:
            x = int(block.get("x_pct", 0) * img_w)
            y = int(block.get("y_pct", 0) * img_h)
            w = int(block.get("w_pct", 1.0) * img_w)
            h = int(block.get("h_pct", 0.05) * img_h)
            page_segments.append({
                "translated_text": block.get("translated_text", ""),
                "bbox": [x, y, w, h],
                "page": page_number,
                "text": block.get("translated_text", ""),
                "align": block.get("align", "left"),
                "bold": block.get("bold", False),
            })
        return page_number, page_segments

    tasks = list(enumerate(image_paths, start=1))
    results: Dict[int, List[Dict]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vision OCR+Translate"):
            page_number, page_segments = future.result()
            results[page_number] = page_segments

    # Return segments sorted by page order
    segments: List[Dict] = []
    for page_number in sorted(results.keys()):
        segments.extend(results[page_number])
    return segments


def attach_translations(
    segments: Sequence[OCRSegment],
    translated_texts: Sequence[str],
) -> List[Dict]:
    output: List[Dict] = []
    for segment, translated in zip(segments, translated_texts):
        payload = segment.to_dict()
        payload["translated_text"] = translated
        output.append(payload)
    return output


def ensure_output_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def get_numbered_output_path(output_path: Path) -> Path:
    if not output_path.exists():
        return output_path
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def summarize_translated_text(translated_segments: List[Dict], translator) -> Dict:
    full_text = "\n".join(
        seg.get("translated_text", "").strip()
        for seg in translated_segments
        if seg.get("translated_text", "").strip()
    )
    if not full_text:
        return {}

    prompt = (
        "You are a legal document analyst. Read the following translated land/legal document text and provide a structured English summary.\n"
        "Rules:\n"
        "- Output ONLY a valid JSON object. No explanations, no markdown, no extra text.\n"
        "- Preserve all legal terminology, names, survey numbers, dates, and amounts exactly.\n"
        "- Use exactly these keys:\n"
        '  "document_type": string\n'
        '  "parties_involved": string\n'
        '  "property_details": string\n'
        '  "key_dates": string\n'
        '  "legal_terms": string\n'
        '  "summary": string - write a detailed, elaborate paragraph explaining the full context of the document, '
        'what it means legally, who is involved, what land or property is affected, what actions were taken, '
        'and any important implications. Minimum 5-6 sentences.\n'
        "- If a section has no information, use an empty string.\n\n"
        f"Document Text:\n{full_text}"
    )

    from google.genai.types import GenerateContentConfig
    json_config = GenerateContentConfig(
        temperature=translator.generation_config.temperature,
        max_output_tokens=translator.generation_config.max_output_tokens,
        response_mime_type="application/json",
    )

    response = translator.client.models.generate_content(
        model=translator.model_name,
        contents=prompt,
        config=json_config,
    )

    # Track summary tokens separately
    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        translator._summary_input_tokens = getattr(translator, "_summary_input_tokens", 0) + input_tokens
        translator._summary_output_tokens = getattr(translator, "_summary_output_tokens", 0) + output_tokens

    import json, re
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    config_dir = config_path.parent

    input_pdf = Path(args.input).expanduser().resolve()
    output_pdf = Path(args.output).expanduser().resolve()
    ensure_output_dir(output_pdf)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    translator_config = config.get("translation", {})
    render_config = config.get("render", {})
    runtime_config = config.get("runtime", {})

    dpi = args.dpi or render_config.get("dpi", 250)
    max_workers = args.max_workers or runtime_config.get("max_workers") or max(1, mp.cpu_count() - 1)
    image_format = render_config.get("image_format", "png")

    use_vision_ocr = config.get("runtime", {}).get("use_vision_ocr", False) and args.lang != "eng"
    output_pdf = get_numbered_output_path(output_pdf)
    LOGGER.info("Output will be saved to %s", output_pdf)
    LOGGER.info("Starting pipeline for %s", input_pdf)

    translator = VertexTranslator(
        project=args.project or translator_config.get("project"),
        credentials_path=resolve_config_relative_path(config_dir, translator_config.get("credentials_path")),
        location=args.location or translator_config.get("location", "us-central1"),
        model_name=args.model or translator_config.get("model"),
        temperature=translator_config.get("temperature", 0.1),
        max_output_tokens=translator_config.get("max_output_tokens", 2048),
        batch_size=translator_config.get("batch_size", 20),
        max_retries=translator_config.get("max_retries", 5),
        retry_base_delay=translator_config.get("retry_base_delay_seconds", 2.0),
    )

    with tempfile.TemporaryDirectory(prefix="pdf_translate_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        image_dir = tmp_dir / "pages"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = save_page_images(input_pdf, image_dir, dpi=dpi, fmt=image_format)

        if use_vision_ocr:
            LOGGER.info("Using Gemini Vision OCR (skipping Tesseract)")
            lang_name = LANG_NAMES.get(args.lang, args.lang)
            is_english = args.lang == "eng"
            if is_english:
                LOGGER.info("Source language is English — Vision OCR will extract text only, no translation.")
            translated_segments = vision_ocr_pages(image_paths, translator, lang_name, max_workers=max_workers, english_only=is_english)
            LOGGER.info("Vision OCR produced %s segments", len(translated_segments))
        else:
            validate_tesseract(config.get("ocr", {}), args.lang)
            page_segments = ocr_pages_in_parallel(image_paths, args.lang, config, max_workers=max_workers)
            segments = flatten(page_segments)
            LOGGER.info("Collected %s OCR text segments", len(segments))

            translated_segments = []
            if segments:
                if args.lang == "eng":
                    LOGGER.info("Source language is English — skipping translation, using OCR text as-is.")
                    translated_segments = [
                        {**seg.to_dict(), "translated_text": seg.text}
                        for seg in segments
                    ]
                else:
                    translation_batches = list(chunk_segments_for_translation(segments, translator.batch_size))
                    for batch in tqdm(translation_batches, total=len(translation_batches), desc="Translating batches"):
                        originals = [segment.text for segment in batch]
                        translated = translator.translate_text_batch(originals)
                        translated_segments.extend(attach_translations(batch, translated))
            else:
                LOGGER.warning("OCR produced no text segments. Rebuilding PDF with original page images only.")

        rebuild_translated_pdf(
            page_image_paths=image_paths,
            translated_segments=translated_segments,
            output_pdf_path=output_pdf,
            rebuild_config=config.get("rebuild", {}),
        )

        if translated_segments:
            LOGGER.info("Generating document summary...")
            summary = summarize_translated_text(translated_segments, translator)
            if summary:
                summary_path = output_pdf.with_suffix(".summary.txt")
                summary_path.write_text(summary, encoding="utf-8")
                LOGGER.info("Summary saved to %s", summary_path)

    LOGGER.info("Finished. Output saved to %s", output_pdf)


if __name__ == "__main__":
    mp.freeze_support()
    main()
