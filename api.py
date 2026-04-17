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
from translate import create_translator, create_summarizer


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


def build_translator(config: dict, config_path: Path):
    return create_translator(config.get("translation", {}), config_path.parent)


def build_summarizer(config: dict, config_path: Path):
    translator_cfg = config.get("translation", {})
    summarization_cfg = config.get("summarization", translator_cfg)
    return create_summarizer(summarization_cfg, config_path.parent)


# Gemini 2.5 Flash pricing (per 1M tokens)
MODEL_PRICING = {
    "gemini-2.5-pro":              {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":            {"input": 0.30,  "output": 2.50},
    "gemini-2.5-flash-lite":       {"input": 0.10,  "output": 0.40},
    "gemini-1.5-flash":            {"input": 0.075, "output": 0.30},
    "gemini-3-flash-preview":      {"input": 0.50,  "output": 3.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
}

def get_token_usage_and_cost(translator, summarizer=None) -> dict:
    trans_in  = getattr(translator, "_translation_input_tokens", 0)
    trans_out = getattr(translator, "_translation_output_tokens", 0)
    summ_obj  = summarizer if summarizer is not None else translator
    summ_in   = getattr(summ_obj, "_summary_input_tokens", 0)
    summ_out  = getattr(summ_obj, "_summary_output_tokens", 0)
    total_in  = trans_in + summ_in
    total_out = trans_out + summ_out

    trans_pricing = MODEL_PRICING.get(translator.model_name, {"input": 0, "output": 0})
    summ_pricing  = MODEL_PRICING.get(summ_obj.model_name, {"input": 0, "output": 0})
    cost_usd = (
        (trans_in * trans_pricing["input"] + trans_out * trans_pricing["output"]) +
        (summ_in  * summ_pricing["input"]  + summ_out  * summ_pricing["output"])
    ) / 1_000_000

    LOGGER.info(
        "Token usage — translation model: %s in: %s out: %s | summary model: %s in: %s out: %s | total: %s | cost: $%.6f",
        translator.model_name, trans_in, trans_out, summ_obj.model_name, summ_in, summ_out, total_in + total_out, cost_usd,
    )
    return {
        "translation_model": translator.model_name,
        "summarization_model": summ_obj.model_name,
        "translation_input_tokens": trans_in,
        "translation_output_tokens": trans_out,
        "summary_input_tokens": summ_in,
        "summary_output_tokens": summ_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": round(cost_usd, 6),
    }


def run_ocr_and_translate(input_pdf: Path, lang: str, config: dict, translator, need_layout: bool = False):
    render_cfg = config.get("render", {})
    dpi = render_cfg.get("dpi", 250)
    image_format = render_cfg.get("image_format", "png")
    # For PDF rebuild we need accurate bounding boxes — keep Vision OCR even for English.
    # For text-only endpoints (summary, translate-text) Tesseract is sufficient and free.
    use_vision_ocr = config.get("runtime", {}).get("use_vision_ocr", False) and (lang != "eng" or need_layout)

    tmp_dir = tempfile.mkdtemp(prefix="pdf_translate_")
    image_dir = Path(tmp_dir) / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = save_page_images(input_pdf, image_dir, dpi=dpi, fmt=image_format)

    is_english = lang == "eng"
    if use_vision_ocr:
        lang_name = LANG_NAMES.get(lang, lang)
        if is_english:
            LOGGER.info("Source language is English — Vision OCR will extract text only, no translation.")
        translated_segments = vision_ocr_pages(image_paths, translator, lang_name, max_workers=4, english_only=is_english)
    else:
        from ocr import validate_tesseract
        validate_tesseract(config.get("ocr", {}), lang)
        page_segments = ocr_pages_in_parallel(image_paths, lang, config, max_workers=4)
        segments = flatten(page_segments)
        translated_segments = []
        if segments:
            if is_english:
                LOGGER.info("Source language is English — skipping translation, using OCR text as-is.")
                translated_segments = [
                    {**seg.to_dict(), "translated_text": seg.text}
                    for seg in segments
                ]
            else:
                from main import chunk_segments_for_translation, detect_page_language
                for page_segs in page_segments:
                    if not page_segs:
                        continue
                    page_text = " ".join(seg.text for seg in page_segs)
                    detected = detect_page_language(page_text)
                    page_num = page_segs[0].page
                    if detected == "en":
                        LOGGER.info("Page %s detected as English — skipping translation.", page_num)
                        translated_segments.extend([
                            {**seg.to_dict(), "translated_text": seg.text}
                            for seg in page_segs
                        ])
                    else:
                        batches = list(chunk_segments_for_translation(page_segs, translator.batch_size))
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
        lines.append(f"{label}:")
        if isinstance(value, list):
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append("" if value is None else str(value))
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

    file_bytes = await file.read()

    # English PDFs are already in the target language — return the original as-is.
    if lang == "eng":
        output_filename = f"translated_{file.filename}"
        output_pdf = OUTPUT_DIR / output_filename
        output_pdf.write_bytes(file_bytes)
        base_url = str(request.base_url).rstrip("/")
        LOGGER.info("English PDF — returning original file as translated output (no rebuild needed).")
        return JSONResponse(content={
            "message": "Document is already in English. Original returned as-is.",
            "download_url": f"{base_url}/download/{output_filename}",
            "filename": output_filename,
            "token_usage": {"model": "none", "translation_input_tokens": 0, "translation_output_tokens": 0,
                            "summary_input_tokens": 0, "summary_output_tokens": 0,
                            "total_tokens": 0, "estimated_cost_usd": 0.0},
        })

    config, config_path = load_config()
    translator = build_translator(config, config_path)
    summarizer = build_summarizer(config, config_path)

    output_filename = f"translated_{file.filename}"
    output_pdf = OUTPUT_DIR / output_filename

    with tempfile.TemporaryDirectory(prefix="api_translate_") as tmp:
        input_pdf = Path(tmp) / file.filename
        input_pdf.write_bytes(file_bytes)

        try:
            image_paths, translated_segments = run_ocr_and_translate(
                input_pdf, lang, config, translator, need_layout=True
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

    token_info = get_token_usage_and_cost(translator, summarizer)
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
    summarizer = build_summarizer(config, config_path)

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
            summary = summarize_translated_text(translated_segments, summarizer)
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
        response_payload["token_usage"] = get_token_usage_and_cost(translator, summarizer)

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
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
    ]

    def run_for_model(model_name: str) -> dict:
        # Build translator and summarizer with overridden model for comparison
        overridden_cfg = {**config.get("translation", {}), "model": model_name}
        translator = create_translator(overridden_cfg, config_path.parent)
        summarizer = create_summarizer({**config.get("summarization", overridden_cfg), "model": model_name}, config_path.parent)

        with tempfile.TemporaryDirectory(prefix=f"api_compare_{model_name}_") as tmp:
            input_pdf = Path(tmp) / file.filename
            input_pdf.write_bytes(file_bytes)
            try:
                _, translated_segments = run_ocr_and_translate(
                    input_pdf, lang, config, translator
                )
                summarize_translated_text(translated_segments, summarizer)
            except Exception as exc:
                LOGGER.exception("Compare failed for model %s", model_name)
                return {"model": model_name, "error": str(exc)}

        return get_token_usage_and_cost(translator, summarizer)

    # Run all models in parallel using threads
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(None, run_for_model, m) for m in COMPARE_MODELS]
    )

    USD_TO_INR = 92.72
    return JSONResponse(content={
        "document": file.filename,
        "language": lang,
        "results": [
            {
                "model": COMPARE_MODELS[i],
                **results[i],
                "estimated_cost_inr": round(results[i].get("estimated_cost_usd", 0) * USD_TO_INR, 4),
            }
            for i in range(len(COMPARE_MODELS))
        ],
    })
