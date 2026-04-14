import asyncio
import html
import html
import json
import logging
import tempfile
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
STATIC_DIR = BASE_DIR / "static"

# Serve files in api_outputs/ as static downloads
app.mount("/download", StaticFiles(directory=str(OUTPUT_DIR)), name="download")

# Serve the frontend UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


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
        model_name=translator_cfg.get("model"),
        temperature=translator_cfg.get("temperature", 0.1),
        max_output_tokens=translator_cfg.get("max_output_tokens", 16384),
        batch_size=translator_cfg.get("batch_size", 20),
        max_retries=translator_cfg.get("max_retries", 5),
        retry_base_delay=translator_cfg.get("retry_base_delay_seconds", 2.0),
    )


# Gemini 2.5 Flash pricing (per 1M tokens)
MODEL_PRICING = {
    "gemini-2.5-pro":              {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":            {"input": 0.30,  "output": 2.50},
    "gemini-2.5-flash-lite":       {"input": 0.10,  "output": 0.40},
    "gemini-1.5-flash":            {"input": 0.075, "output": 0.30},
    "gemini-3-flash-preview":      {"input": 0.50,  "output": 3.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
}

def get_token_usage_and_cost(translator: VertexTranslator) -> dict:
    trans_in  = getattr(translator, "_translation_input_tokens", 0)
    trans_out = getattr(translator, "_translation_output_tokens", 0)
    summ_in   = getattr(translator, "_summary_input_tokens", 0)
    summ_out  = getattr(translator, "_summary_output_tokens", 0)
    total_in  = trans_in + summ_in
    total_out = trans_out + summ_out
    pricing   = MODEL_PRICING.get(translator.model_name, {"input": 0, "output": 0})
    cost_usd  = (total_in * pricing["input"] + total_out * pricing["output"]) / 1_000_000
    LOGGER.info(
        "Token usage — model: %s | translation in: %s out: %s | summary in: %s out: %s | total: %s | cost: $%.6f",
        translator.model_name, trans_in, trans_out, summ_in, summ_out, total_in + total_out, cost_usd,
    )
    return {
        "model": translator.model_name,
        "translation_input_tokens": trans_in,
        "translation_output_tokens": trans_out,
        "summary_input_tokens": summ_in,
        "summary_output_tokens": summ_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": round(cost_usd, 6),
    }


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


def build_summary_html(summary: dict) -> str:
    pretty_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    sections = []
    for key, value in summary.items():
        label = html.escape(key.replace("_", " ").title())
        content = html.escape("" if value is None else str(value))
        sections.append(
            f"<section><h2>{label}</h2><div class='value'>{content}</div></section>"
        )

    body = "\n".join(sections) if sections else "<p>No summary content available.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Summary</title>
  <style>
    body {{
      margin: 0;
      background: #f3f4f6;
      color: #1f2937;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .page {{
      max-width: 900px;
      margin: 32px auto;
      background: #fff;
      padding: 40px 48px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
      border: 1px solid #e5e7eb;
    }}
    h1 {{
      margin: 0 0 24px;
      font-size: 32px;
      border-bottom: 2px solid #111827;
      padding-bottom: 12px;
    }}
    h2 {{
      margin: 24px 0 8px;
      font-size: 18px;
    }}
    .value {{
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 16px;
    }}
    details {{
      margin-top: 28px;
      border-top: 1px solid #e5e7eb;
      padding-top: 20px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      padding: 16px;
      overflow: auto;
      font-size: 13px;
    }}
    @media print {{
      body {{ background: white; }}
      .page {{
        margin: 0;
        max-width: none;
        box-shadow: none;
        border: none;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Document Summary</h1>
    {body}
    <details>
      <summary>Raw JSON</summary>
      <pre>{pretty_json}</pre>
    </details>
  </main>
</body>
</html>"""


def build_summary_text(summary: dict) -> str:
    lines = ["Document Summary", "=" * 16, ""]
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        content = "" if value is None else str(value)
        lines.append(f"{label}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@app.post("/translate-text")
async def translate_pdf_text(
    file: UploadFile = File(...),
    lang: str = Form("hin"),
):
    """
    Upload a scanned PDF and get back the translated text as JSON (no PDF rebuild).
    lang: hin | kan | tam | tel | eng
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    config, config_path = load_config()
    translator = build_translator(config, config_path)

    with tempfile.TemporaryDirectory(prefix="api_translate_text_") as tmp:
        input_pdf = Path(tmp) / file.filename
        input_pdf.write_bytes(await file.read())

        try:
            _, translated_segments = run_ocr_and_translate(
                input_pdf, lang, config, translator
            )
        except Exception as exc:
            LOGGER.exception("Translation (text) failed")
            raise HTTPException(status_code=500, detail=str(exc))

    # Group segments by page, ordered by vertical position
    pages: dict = {}
    for seg in translated_segments:
        page_num = int(seg.get("page", 1))
        text = (seg.get("translated_text") or "").strip()
        if text:
            pages.setdefault(page_num, []).append((seg.get("bbox", [0, 0, 0, 0]), text))

    result_pages = []
    for page_num in sorted(pages.keys()):
        segments = sorted(pages[page_num], key=lambda s: (s[0][1], s[0][0]))  # sort by y then x
        page_text = "\n".join(text for _, text in segments)
        result_pages.append({"page": page_num, "text": page_text})

    return JSONResponse(content={"pages": result_pages})


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

    token_info = get_token_usage_and_cost(translator)
    return JSONResponse(content={
        "message": "Translation successful",
        "download_url": download_url,
        "filename": output_filename,
        "token_usage": token_info,
    })


@app.post("/summary")
async def summarize_pdf(
    request: Request,
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

    summary_txt_filename = f"summary_{Path(file.filename).stem}.txt"
    summary_json_filename = f"summary_{Path(file.filename).stem}.json"
    summary_html_filename = f"summary_{Path(file.filename).stem}.html"
    summary_txt_path = OUTPUT_DIR / summary_txt_filename
    summary_json_path = OUTPUT_DIR / summary_json_filename
    summary_html_path = OUTPUT_DIR / summary_html_filename

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

        summary_txt_path.write_text(build_summary_text(summary), encoding="utf-8")
        summary_json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_html_path.write_text(build_summary_html(summary), encoding="utf-8")

        base_url = str(request.base_url).rstrip("/")
        response_payload = dict(summary)
        response_payload["url_summary"] = f"{base_url}/download/{summary_txt_filename}"
        response_payload["token_usage"] = get_token_usage_and_cost(translator)

        return JSONResponse(content=response_payload)


@app.post("/compare")
async def compare_models(
    file: UploadFile = File(...),
    lang: str = Form("hin"),
):
    """
    Run the same document through both gemini-2.5-pro and gemini-2.5-flash in parallel.
    Returns token usage and estimated cost for both models — no PDF output.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    config, config_path = load_config()

    COMPARE_MODELS = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ]

    def run_for_model(model_name: str) -> dict:
        # Build a translator with overridden model
        translator_cfg = config.get("translation", {})
        config_dir = config_path.parent
        translator = VertexTranslator(
            project=translator_cfg.get("project"),
            credentials_path=resolve_config_relative_path(config_dir, translator_cfg.get("credentials_path")),
            location=translator_cfg.get("location", "us-central1"),
            model_name=model_name,
            temperature=translator_cfg.get("temperature", 0.1),
            max_output_tokens=translator_cfg.get("max_output_tokens", 16384),
            batch_size=translator_cfg.get("batch_size", 20),
            max_retries=translator_cfg.get("max_retries", 5),
            retry_base_delay=translator_cfg.get("retry_base_delay_seconds", 2.0),
        )

        with tempfile.TemporaryDirectory(prefix=f"api_compare_{model_name}_") as tmp:
            input_pdf = Path(tmp) / file.filename
            input_pdf.write_bytes(file_bytes)
            try:
                _, translated_segments = run_ocr_and_translate(
                    input_pdf, lang, config, translator
                )
                summarize_translated_text(translated_segments, translator)
            except Exception as exc:
                LOGGER.exception("Compare failed for model %s", model_name)
                return {"model": model_name, "error": str(exc)}

        return get_token_usage_and_cost(translator)

    # Run all models in parallel using threads
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(None, run_for_model, m) for m in COMPARE_MODELS]
    )

    return JSONResponse(content={
        "document": file.filename,
        "language": lang,
        "results": [
            {"model": COMPARE_MODELS[i], **results[i]} for i in range(len(COMPARE_MODELS))
        ],
    })
