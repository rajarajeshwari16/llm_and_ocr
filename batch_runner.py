"""
batch_runner.py
---------------
Runs all 13 land documents through:
  - Translation: 3 models (gemini-2.5-pro, gemini-2.5-flash, gemini-3-flash-preview)
  - Summary: direct summarization only (passthrough translation)

Outputs: batch_results.csv — ready to paste into Excel.
Does NOT modify any existing files.
"""

import csv
import sys
import tempfile
import logging
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from main import (
    save_page_images,
    vision_ocr_pages,
    ocr_pages_in_parallel,
    flatten,
    attach_translations,
    LANG_NAMES,
    resolve_config_path,
    chunk_segments_for_translation,
    detect_page_language,
)
from translate import create_translator, create_summarizer, PassthroughTranslator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("batch_runner")

BASE_DIR   = Path(__file__).resolve().parent
DOCS_DIR   = BASE_DIR / "land_document"
OUTPUT_CSV = BASE_DIR / "batch_results.csv"

# Language mapping per document
DOCUMENTS = [
    {"name": "39_BAL_MR",                                          "file": "kannada-doc.pdf",                                   "lang": "kan"},
    {"name": "44_Dehradun_Khatauni_Latest",                        "file": "44_Dehradun_Khatauni_Latest.pdf",                   "lang": "hin"},
    {"name": "47_Tonk_Jamabandi_Khasra No.12",                     "file": "47_Tonk_Jamabandi_Khasra No.12.pdf",               "lang": "hin"},
    {"name": "55_Bhopal_Campus_Lease Deed_MP059712021A1084857",    "file": "55_Bhopal_Campus_Lease Deed_MP059712021A1084857.pdf","lang": "hin"},
    {"name": "55_Bhopal_Campus_OTH_Nazul Patta-Agreement",        "file": "55_Bhopal_Campus_OTH_Nazul Patta-Agreement.pdf",    "lang": "hin"},
    {"name": "C16_BLR_APU_818-2012 Judgment",                      "file": "C16_BLR_APU_818-2012 Judgment.pdf",                "lang": "eng"},
    {"name": "C16_BLR_APU_OS 546-1996 Judgement",                  "file": "C16_BLR_APU_OS 546-1996 Judgement.pdf",            "lang": "eng"},
    {"name": "C16_BLR_APU_OS 818-2012 Deposition",                 "file": "C16_BLR_APU_OS 818-2012 Deposition.pdf",           "lang": "kan"},
    {"name": "OS 975 Notice from Tehsil-27-3-25",                  "file": "OS 975 Notice from Tehsil-27-3-25.pdf",            "lang": "hin"},
    {"name": "OS 975_OTH_Letter to Tehsil-17-04-2025",             "file": "OS 975_OTH_Letter to Tehsil-17-04-2025.pdf",       "lang": "hin"},
    {"name": "OS 975_OTH_Letter to Tehsil-28-3-25",                "file": "OS 975_OTH_Letter to Tehsil-28-3-25.pdf",          "lang": "hin"},
    {"name": "OS 975_Plaint copy-Preetam Prasad Rawat",            "file": "OS 975_Plaint copy-Preetam Prasad Rawat.pdf",      "lang": "hin"},
    {"name": "OS_975_OTH_Reply for SDM",                           "file": "OS_975_OTH_Reply for SDM.pdf",                     "lang": "hin"},
]

MODELS = [
    {"name": "gemini-2.5-pro",          "location": "us-central1"},
    {"name": "gemini-2.5-flash",        "location": "us-central1"},
    {"name": "gemini-3-flash-preview",  "location": "global"},
]


def load_config():
    config_path = resolve_config_path(None)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_path


def ocr_and_translate(input_pdf: Path, lang: str, config: dict, translator):
    render_cfg  = config.get("render", {})
    runtime_cfg = config.get("runtime", {})
    dpi         = render_cfg.get("dpi", 250)
    fmt         = render_cfg.get("image_format", "png")
    max_workers = runtime_cfg.get("max_workers", 2)
    use_vision  = runtime_cfg.get("use_vision_ocr", False) and lang != "eng"

    with tempfile.TemporaryDirectory(prefix="batch_") as tmp:
        image_dir = Path(tmp) / "pages"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = save_page_images(input_pdf, image_dir, dpi=dpi, fmt=fmt)

        is_passthrough = isinstance(translator, PassthroughTranslator)

        # When passthrough + vision OCR needed: use summarizer config to extract text
        ocr_extractor = translator
        if use_vision and is_passthrough:
            summ_cfg = config.get("summarization", config.get("translation", {}))
            ocr_extractor = create_translator({**summ_cfg, "provider": "vertex_gemini"}, BASE_DIR)

        if use_vision:
            lang_name       = LANG_NAMES.get(lang, lang)
            extract_verbatim = lang == "eng" or is_passthrough
            segments = vision_ocr_pages(image_paths, ocr_extractor, lang_name,
                                        max_workers=max_workers, english_only=extract_verbatim)
        else:
            from ocr import validate_tesseract
            validate_tesseract(config.get("ocr", {}), lang)
            page_segs_list = ocr_pages_in_parallel(image_paths, lang, config, max_workers=max_workers)
            flat = flatten(page_segs_list)
            segments = []
            if flat:
                if lang == "eng" or is_passthrough:
                    segments = [{**s.to_dict(), "translated_text": s.text} for s in flat]
                else:
                    for page_segs in page_segs_list:
                        if not page_segs:
                            continue
                        page_text = " ".join(s.text for s in page_segs)
                        if detect_page_language(page_text) == "en":
                            segments.extend([{**s.to_dict(), "translated_text": s.text} for s in page_segs])
                        else:
                            batches = list(chunk_segments_for_translation(page_segs, translator.batch_size))
                            for batch in batches:
                                translated = translator.translate_text_batch([s.text for s in batch])
                                segments.extend(attach_translations(batch, translated))
        return segments


def run_translation_for_model(doc: dict, model: dict, config: dict) -> dict:
    """Run translation only — capture translation tokens."""
    model_name = model["name"]
    location   = model["location"]
    LOGGER.info("[TRANSLATION] %s | model=%s | location=%s", doc["name"], model_name, location)
    trans_cfg  = {**config.get("translation", {}), "model": model_name, "provider": "vertex_gemini", "location": location}
    translator = create_translator(trans_cfg, BASE_DIR)

    input_pdf = DOCS_DIR / doc["file"]
    try:
        ocr_and_translate(input_pdf, doc["lang"], config, translator)
    except Exception as exc:
        LOGGER.error("Translation failed: %s", exc)
        return {"translation_input_tokens": "ERROR", "translation_output_tokens": "ERROR"}

    return {
        "translation_input_tokens":  getattr(translator, "_translation_input_tokens", 0),
        "translation_output_tokens": getattr(translator, "_translation_output_tokens", 0),
    }


def run_direct_summary_for_model(doc: dict, model: dict, config: dict) -> dict:
    """Run direct summarization (passthrough translation) — capture summary tokens."""
    model_name = model["name"]
    location   = model["location"]
    LOGGER.info("[SUMMARY] %s | model=%s | location=%s", doc["name"], model_name, location)
    trans_cfg  = {**config.get("translation", {}), "provider": "passthrough"}
    summ_cfg   = {**config.get("summarization", {}), "model": model_name, "location": location}
    translator = create_translator(trans_cfg, BASE_DIR)
    summarizer = create_summarizer(summ_cfg, BASE_DIR)

    input_pdf = DOCS_DIR / doc["file"]
    try:
        segments = ocr_and_translate(input_pdf, doc["lang"], config, translator)
        if segments:
            summarizer.summarize(segments)
    except Exception as exc:
        LOGGER.error("Summary failed: %s", exc)
        return {"summary_input_tokens": "ERROR", "summary_output_tokens": "ERROR"}

    return {
        "summary_input_tokens":  getattr(summarizer, "_summary_input_tokens", 0),
        "summary_output_tokens": getattr(summarizer, "_summary_output_tokens", 0),
    }


def main():
    config, _ = load_config()

    rows = []

    for doc in DOCUMENTS:
        LOGGER.info("=" * 60)
        LOGGER.info("Processing: %s", doc["name"])
        row = {"Document": doc["name"]}

        for model in MODELS:
            m = model["name"]
            # Translation tokens
            trans_result = run_translation_for_model(doc, model, config)
            row[f"{m}_trans_input"]  = trans_result.get("translation_input_tokens", 0)
            row[f"{m}_trans_output"] = trans_result.get("translation_output_tokens", 0)

            # Summary tokens (direct)
            summ_result = run_direct_summary_for_model(doc, model, config)
            row[f"{m}_summ_input"]   = summ_result.get("summary_input_tokens", 0)
            row[f"{m}_summ_output"]  = summ_result.get("summary_output_tokens", 0)

        rows.append(row)
        LOGGER.info("Done: %s", doc["name"])

    # Write CSV
    fieldnames = ["Document"]
    for model in MODELS:
        m = model["name"]
        fieldnames += [
            f"{m}_trans_input",
            f"{m}_trans_output",
            f"{m}_summ_input",
            f"{m}_summ_output",
        ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Results saved to %s", OUTPUT_CSV)
    print(f"\nDone! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
