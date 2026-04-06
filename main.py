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
    parser.add_argument("--lang", required=True, choices=["hin", "kan", "tam", "tel"], help="Tesseract OCR language code.")
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
    pil_images = convert_from_path(str(input_pdf), dpi=dpi, fmt=fmt)
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

    LOGGER.info("Starting pipeline for %s", input_pdf)
    validate_tesseract(config.get("ocr", {}), args.lang)

    with tempfile.TemporaryDirectory(prefix="pdf_translate_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        image_dir = tmp_dir / "pages"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = save_page_images(input_pdf, image_dir, dpi=dpi, fmt=image_format)
        page_segments = ocr_pages_in_parallel(image_paths, args.lang, config, max_workers=max_workers)
        segments = flatten(page_segments)

        LOGGER.info("Collected %s OCR text segments", len(segments))

        translated_segments: List[Dict] = []
        if segments:
            translator = VertexTranslator(
                project=args.project or translator_config.get("project"),
                credentials_path=resolve_config_relative_path(config_dir, translator_config.get("credentials_path")),
                location=args.location or translator_config.get("location", "us-central1"),
                model_name=args.model or translator_config.get("model", "gemini-1.5-pro"),
                temperature=translator_config.get("temperature", 0.1),
                max_output_tokens=translator_config.get("max_output_tokens", 2048),
                batch_size=translator_config.get("batch_size", 20),
                max_retries=translator_config.get("max_retries", 5),
                retry_base_delay=translator_config.get("retry_base_delay_seconds", 2.0),
            )
            translation_batches = list(chunk_segments_for_translation(segments, translator.batch_size))
            for batch in tqdm(
                translation_batches,
                total=len(translation_batches),
                desc="Translating batches",
            ):
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

    LOGGER.info("Finished. Output saved to %s", output_pdf)


if __name__ == "__main__":
    mp.freeze_support()
    main()
