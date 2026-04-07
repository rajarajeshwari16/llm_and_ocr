import logging
import tempfile
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from main import (
    attach_translations,
    flatten,
    ocr_pages_in_parallel,
    resolve_config_path,
    resolve_config_relative_path,
    save_page_images,
    summarize_translated_text,
    vision_ocr_pages,
    LANG_NAMES,
)
from rebuild import rebuild_translated_pdf
from translate import VertexTranslator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("pdf_translate.api")

app = FastAPI(title="PDF Translate API")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "api_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve files in api_outputs/ as static downloads
app.mount("/download", StaticFiles(directory=str(OUTPUT_DIR)), name="download")


def load_config():
    config_path = resolve_config_path(None)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_path


def build_translator(config: dict, config_path: Path) -> VertexTranslator:
    translator_cfg = config.get("translation", {})
    config_dir = config_path.parent
    return VertexTranslator(
        project=translator_cfg.get("project"),
        credentials_path=resolve_config_relative_path(config_dir, translator_cfg.get("credentials_path")),
        location=translator_cfg.get("location", "us-central1"),
        model_name=translator_cfg.get("model", "gemini-2.5-pro"),
        temperature=translator_cfg.get("temperature", 0.1),
        max_output_tokens=translator_cfg.get("max_output_tokens", 16384),
        batch_size=translator_cfg.get("batch_size", 20),
        max_retries=translator_cfg.get("max_retries", 5),
        retry_base_delay=translator_cfg.get("retry_base_delay_seconds", 2.0),
    )


def run_ocr_and_translate(input_pdf: Path, lang: str, config: dict, translator: VertexTranslator):
    render_cfg = config.get("render", {})
    dpi = render_cfg.get("dpi", 250)
    image_format = render_cfg.get("image_format", "png")
    use_vision_ocr = config.get("runtime", {}).get("use_vision_ocr", False)

    tmp_dir = tempfile.mkdtemp(prefix="pdf_translate_")
    image_dir = Path(tmp_dir) / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = save_page_images(input_pdf, image_dir, dpi=dpi, fmt=image_format)

    if use_vision_ocr:
        lang_name = LANG_NAMES.get(lang, lang)
        translated_segments = vision_ocr_pages(image_paths, translator, lang_name, max_workers=4)
    else:
        from ocr import validate_tesseract
        validate_tesseract(config.get("ocr", {}), lang)
        page_segments = ocr_pages_in_parallel(image_paths, lang, config, max_workers=4)
        segments = flatten(page_segments)
        translated_segments = []
        if segments:
            from main import chunk_segments_for_translation
            batches = list(chunk_segments_for_translation(segments, translator.batch_size))
            for batch in batches:
                originals = [s.text for s in batch]
                translated = translator.translate_text_batch(originals)
                translated_segments.extend(attach_translations(batch, translated))

    return image_paths, translated_segments


@app.post("/translate")
async def translate_pdf(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Form("hin"),
):
    """
    Upload a scanned PDF and get back a download URL for the translated PDF.
    lang: hin | kan | tam | tel | eng
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    config, config_path = load_config()
    translator = build_translator(config, config_path)

    output_filename = f"translated_{file.filename}"
    output_pdf = OUTPUT_DIR / output_filename

    with tempfile.TemporaryDirectory(prefix="api_translate_") as tmp:
        input_pdf = Path(tmp) / file.filename
        input_pdf.write_bytes(await file.read())

        try:
            image_paths, translated_segments = run_ocr_and_translate(
                input_pdf, lang, config, translator
            )
            rebuild_translated_pdf(
                page_image_paths=image_paths,
                translated_segments=translated_segments,
                output_pdf_path=output_pdf,
                rebuild_config=config.get("rebuild", {}),
            )
        except Exception as exc:
            LOGGER.exception("Translation failed")
            raise HTTPException(status_code=500, detail=str(exc))

    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/{output_filename}"

    return JSONResponse(content={
        "message": "Translation successful",
        "download_url": download_url,
        "filename": output_filename,
    })


@app.post("/summary")
async def summarize_pdf(
    file: UploadFile = File(...),
    lang: str = Form("hin"),
):
    """
    Upload a scanned PDF and get back a structured English summary.
    lang: hin | kan | tam | tel | eng
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    config, config_path = load_config()
    translator = build_translator(config, config_path)

    with tempfile.TemporaryDirectory(prefix="api_summary_") as tmp:
        input_pdf = Path(tmp) / file.filename
        input_pdf.write_bytes(await file.read())

        try:
            _, translated_segments = run_ocr_and_translate(
                input_pdf, lang, config, translator
            )
            summary = summarize_translated_text(translated_segments, translator)
        except Exception as exc:
            LOGGER.exception("Summary generation failed")
            raise HTTPException(status_code=500, detail=str(exc))

        return JSONResponse(content=summary)
