import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import google.auth
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions


LOGGER = logging.getLogger("pdf_translate.translate")


def chunk_segments_for_translation(segments: Sequence, batch_size: int) -> Iterable[Sequence]:
    for index in range(0, len(segments), batch_size):
        yield segments[index : index + batch_size]


# ---------------------------------------------------------------------------
# Abstract base — all translators must implement these three methods
# ---------------------------------------------------------------------------

class BaseTranslator(ABC):
    @abstractmethod
    def translate_text_batch(self, text_list: List[str]) -> List[str]: ...

    @abstractmethod
    def translate_single_text(self, text: str) -> str: ...

    @abstractmethod
    def ocr_translate_page_image(self, image_path: str, page_number: int, lang_name: str, english_only: bool = False) -> list: ...


# ---------------------------------------------------------------------------
# Passthrough — no-op translator, returns text as-is (for English docs)
# ---------------------------------------------------------------------------

class PassthroughTranslator(BaseTranslator):
    def __init__(self):
        self.model_name = "passthrough"
        self._translation_input_tokens = 0
        self._translation_output_tokens = 0

    def translate_text_batch(self, text_list: List[str]) -> List[str]:
        return text_list

    def translate_single_text(self, text: str) -> str:
        return text

    def ocr_translate_page_image(self, image_path: str, page_number: int, lang_name: str, english_only: bool = False) -> list:
        return []


# ---------------------------------------------------------------------------
# Factory — create translator from config
# ---------------------------------------------------------------------------

def create_translator(translation_cfg: dict, config_dir: Path) -> BaseTranslator:
    from main import resolve_config_relative_path
    provider = translation_cfg.get("provider", "vertex_gemini")
    if provider == "passthrough":
        LOGGER.info("Translation provider: passthrough (no-op)")
        return PassthroughTranslator()
    if provider == "vertex_gemini":
        LOGGER.info("Translation provider: vertex_gemini (model=%s)", translation_cfg.get("model"))
        return VertexTranslator(
            project=translation_cfg.get("project"),
            credentials_path=resolve_config_relative_path(config_dir, translation_cfg.get("credentials_path")),
            location=translation_cfg.get("location", "us-central1"),
            model_name=translation_cfg.get("model"),
            temperature=translation_cfg.get("temperature", 0.1),
            max_output_tokens=translation_cfg.get("max_output_tokens", 16384),
            batch_size=translation_cfg.get("batch_size", 20),
            max_retries=translation_cfg.get("max_retries", 5),
            retry_base_delay=translation_cfg.get("retry_base_delay_seconds", 2.0),
        )
    raise ValueError(f"Unknown translation provider: '{provider}'. Valid options: vertex_gemini, passthrough")


# ---------------------------------------------------------------------------
# VertexSummarizer — separate summarization service with its own config
# ---------------------------------------------------------------------------

class VertexSummarizer:
    def __init__(
        self,
        project: str,
        credentials_path: str,
        location: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        retry_base_delay: float,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._summary_input_tokens = 0
        self._summary_output_tokens = 0

        if credentials_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            resolved = Path(credentials_path).expanduser().resolve()
            if resolved.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)

        credentials, detected_project = google.auth.default()
        resolved_project = project or detected_project
        self.client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        self.generation_config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )

    def summarize(self, translated_segments: List[Dict]) -> Dict:
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
            '  "parties_involved": array of strings - each party as a separate item. Format: ["VENDOR: details", "PURCHASER: details"]\n'
            '  "property_details": array of strings - each distinct property detail or boundary as a separate item. Format: ["Survey No: ...", "Extent: ...", "Boundaries: ..."]\n'
            '  "key_dates": array of strings - each date as a separate item. Format: ["12-10-2011: Deed execution", "18-07-2005: Vendor acquisition"]\n'
            '  "legal_terms": array of strings - each legal term or clause as a separate item. Format: ["Absolute Sale Deed", "Occupancy Rights", ...]\n'
            '  "summary": string - write a detailed, elaborate paragraph explaining the full context of the document, '
            'what it means legally, who is involved, what land or property is affected, what actions were taken, '
            'and any important implications. Minimum 5-6 sentences.\n'
            "- If a section has no information, use an empty array [].\n\n"
            f"Document Text:\n{full_text}"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    self._summary_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    self._summary_output_tokens += getattr(usage, "candidates_token_count", 0) or 0
                raw = (response.text or "").strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
                return json.loads(raw.strip())
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Summarization failed after {self.max_retries} attempts.") from exc
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                LOGGER.warning("Summarization attempt %s/%s failed: %s. Retrying in %.1f s.", attempt, self.max_retries, exc, delay)
                time.sleep(delay)
        return {}


def create_summarizer(summarization_cfg: dict, config_dir: Path) -> VertexSummarizer:
    from main import resolve_config_relative_path
    provider = summarization_cfg.get("provider", "vertex_gemini")
    if provider != "vertex_gemini":
        raise ValueError(f"Unknown summarization provider: '{provider}'. Valid options: vertex_gemini")
    LOGGER.info("Summarization provider: vertex_gemini (model=%s)", summarization_cfg.get("model"))
    return VertexSummarizer(
        project=summarization_cfg.get("project"),
        credentials_path=resolve_config_relative_path(config_dir, summarization_cfg.get("credentials_path")),
        location=summarization_cfg.get("location", "us-central1"),
        model_name=summarization_cfg.get("model", "gemini-2.5-flash"),
        temperature=summarization_cfg.get("temperature", 0.1),
        max_output_tokens=summarization_cfg.get("max_output_tokens", 16384),
        max_retries=summarization_cfg.get("max_retries", 5),
        retry_base_delay=summarization_cfg.get("retry_base_delay_seconds", 2.0),
    )


# ---------------------------------------------------------------------------
# VertexTranslator — existing implementation, now extends BaseTranslator
# ---------------------------------------------------------------------------

class VertexTranslator(BaseTranslator):
    def __init__(
        self,
        project: str | None,
        credentials_path: str | None,
        location: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        batch_size: int,
        max_retries: int,
        retry_base_delay: float,
    ) -> None:
        self._ensure_credentials_env(credentials_path)

        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. Point it to your Google Cloud service account JSON file."
            )

        credentials, detected_project = google.auth.default()
        resolved_project = project or detected_project
        if not resolved_project:
            raise EnvironmentError(
                "Could not determine Google Cloud project ID from Application Default Credentials. "
                "Set it in config.yaml or pass --project."
            )

        self.project = resolved_project
        self.location = location
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        self.generation_config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _ensure_credentials_env(self, credentials_path: str | None) -> None:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return

        if not credentials_path:
            return

        resolved_path = Path(credentials_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configured credentials JSON not found: {resolved_path}")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved_path)
        LOGGER.info("Using Google credentials JSON from %s", resolved_path)

    def translate_text_batch(self, text_list: List[str]) -> List[str]:
        if not text_list:
            return []

        prompt = self._build_prompt(text_list)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )
                translated = self._parse_response(response.text, expected_count=len(text_list))
                return translated
            except Exception as exc:
                if len(text_list) > 1 and self._should_fallback_to_single_item(exc):
                    LOGGER.warning(
                        "Batch translation response was not parseable for %s items. Falling back to single-item translation.",
                        len(text_list),
                    )
                    return [self.translate_single_text(text) for text in text_list]

                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Vertex AI translation failed after {self.max_retries} attempts."
                    ) from exc

                delay = self.retry_base_delay * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Translation batch failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        return text_list

    def translate_single_text(self, text: str) -> str:
        prompt = (
            "You are a legal document translator. Translate the following text from an Indian regional language to English.\n"
            "Rules:\n"
            "- Output ONLY the translated English text. Nothing else.\n"
            "- Do NOT add explanations, commentary, notes, or disclaimers.\n"
            "- Do NOT say 'Based on the provided text' or 'This appears to be'.\n"
            "- Preserve all legal terminology, names, numbers, and dates exactly.\n"
            "- If a word is unclear, transliterate it rather than guessing.\n\n"
            f"{text}"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )
                translated = (response.text or "").strip()
                if translated.startswith("```"):
                    translated = translated.strip("`").strip()
                    if translated.lower().startswith("text"):
                        translated = translated[4:].strip()
                if not translated:
                    raise ValueError("Empty translation response.")
                return translated
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Vertex AI single-item translation failed after {self.max_retries} attempts."
                    ) from exc

                delay = self.retry_base_delay * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Single-item translation failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        return text

    def ocr_translate_page_image(
        self,
        image_path: str,
        page_number: int,
        lang_name: str,
        english_only: bool = False,
    ) -> list:
        from pathlib import Path as _Path
        from google.genai import types as genai_types

        image_bytes = _Path(image_path).read_bytes()
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

        if english_only:
            prompt = (
                "You are analyzing a scanned English legal/land document image.\n"
                "Extract every visible text block exactly as it appears. Do NOT translate or modify any text.\n"
                "Return ONLY a valid JSON array. No markdown, no explanation, nothing else.\n"
                "Format:\n"
                '[\n'
                '  {"translated_text": "text exactly as it appears", "x_pct": 0.05, "y_pct": 0.02, "w_pct": 0.9, "h_pct": 0.04, "align": "center", "bold": true},\n'
                '  ...\n'
                ']\n'
                "Rules:\n"
                "- x_pct, y_pct = top-left corner as fraction of image width/height (0.0 to 1.0)\n"
                "- w_pct, h_pct = width/height as fraction of image dimensions (0.0 to 1.0)\n"
                "- align: detect text alignment from the image - use 'left', 'center', or 'right'\n"
                "- bold: true if the text appears bold or underlined in the image, false otherwise\n"
                "- Preserve all names, numbers, survey numbers, dates, and legal terms exactly\n"
                "- Include every text block including table headers and cells\n"
                "- Output ONLY the JSON array\n"
                "- Copy the text verbatim — do NOT translate, paraphrase, or alter it in any way."
            )
        else:
            prompt = (
                f"You are analyzing a scanned {lang_name} legal/land document image.\n"
                "Extract every visible text block and translate each one to English.\n"
                "Return ONLY a valid JSON array. No markdown, no explanation, nothing else.\n"
                "Format:\n"
                '[\n'
                '  {"translated_text": "English text here", "x_pct": 0.05, "y_pct": 0.02, "w_pct": 0.9, "h_pct": 0.04, "align": "center", "bold": true},\n'
                '  ...\n'
                ']\n'
                "Rules:\n"
                "- x_pct, y_pct = top-left corner as fraction of image width/height (0.0 to 1.0)\n"
                "- w_pct, h_pct = width/height as fraction of image dimensions (0.0 to 1.0)\n"
                "- align: detect text alignment from the image - use 'left', 'center', or 'right'\n"
                "- bold: true if the text appears bold or underlined in the image, false otherwise\n"
                "- Preserve all names, numbers, survey numbers, dates, and legal terms exactly\n"
                "- Include every text block including table headers and cells\n"
                "- Do NOT add explanations or commentary\n"
                "- Output ONLY the JSON array\n"
                "- CRITICAL: translated_text must ALWAYS be in English only. Never return the original language text.\n"
                "- If text is already in English, keep it as-is in English.\n"
                "- If text is in any Indian language, you MUST translate it to English. Never copy the original script."
            )

        json_config = GenerateContentConfig(
            temperature=self.generation_config.temperature,
            max_output_tokens=self.generation_config.max_output_tokens,
            response_mime_type="application/json",
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                text_part = genai_types.Part.from_text(text=prompt)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[image_part, text_part],
                    config=json_config,
                )
                raw = (response.text or "").strip()
                # Log token usage — handle different field names across model versions
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = (
                        getattr(usage, "prompt_token_count", None) or
                        getattr(usage, "input_token_count", None) or
                        getattr(usage, "total_input_tokens", None) or 0
                    )
                    output_tokens = (
                        getattr(usage, "candidates_token_count", None) or
                        getattr(usage, "output_token_count", None) or
                        getattr(usage, "total_output_tokens", None) or 0
                    )
                    LOGGER.info(
                        "Vision OCR page %s token usage — input: %s, output: %s, total: %s | fields: %s",
                        page_number, input_tokens, output_tokens, input_tokens + output_tokens,
                        [a for a in dir(usage) if not a.startswith("_")],
                    )
                    self._translation_input_tokens = getattr(self, "_translation_input_tokens", 0) + input_tokens
                    self._translation_output_tokens = getattr(self, "_translation_output_tokens", 0) + output_tokens
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()
                blocks = json.loads(raw)
                LOGGER.info("Vision OCR page %s: extracted %s blocks", page_number, len(blocks))
                return blocks
            except Exception as exc:
                if attempt == self.max_retries:
                    LOGGER.error("Vision OCR failed for page %s after %s attempts: %s", page_number, self.max_retries, exc)
                    return []
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Vision OCR attempt %s/%s failed for page %s: %s. Retrying in %.1f s.",
                    attempt, self.max_retries, page_number, exc, delay,
                )
                time.sleep(delay)
        return []

    def _build_prompt(self, text_list: Sequence[str]) -> str:
        blocks = []
        for index, text in enumerate(text_list, start=1):
            blocks.append(
                {
                    "id": index,
                    "prompt": "Translate the following text to English. Preserve meaning. Keep formatting simple:",
                    "text": text,
                }
            )

        instructions = (
            "You are a legal document translator specializing in Indian regional-language scanned PDFs.\n"
            "Translate each input text block into English.\n"
            "Rules:\n"
            "- Output ONLY the translated text in the exact block format below. Nothing else.\n"
            "- Do NOT add explanations, commentary, notes, or disclaimers.\n"
            "- Do NOT say 'Based on the provided text' or 'This appears to be'.\n"
            "- Preserve all legal terminology, names, numbers, and dates exactly.\n"
            "- Keep the number of translations identical to the input. Do not omit entries.\n"
            "- If text is unclear OCR noise, transliterate rather than explain.\n\n"
            "Return the output in this exact block format and nothing else:\n"
            "<<<ID:1>>>\n"
            "translated text here\n"
            "<<<END>>>\n"
            "<<<ID:2>>>\n"
            "translated text here\n"
            "<<<END>>>\n\n"
            f"Inputs:\n{json.dumps(blocks, ensure_ascii=False, indent=2)}"
        )
        return instructions

    def _parse_response(self, raw_text: str, expected_count: int) -> List[str]:
        block_matches = re.findall(
            r"<<<ID:(\d+)>>>\s*(.*?)\s*<<<END>>>",
            raw_text,
            flags=re.DOTALL,
        )
        if block_matches:
            parsed_blocks = {int(match_id): text.strip() for match_id, text in block_matches}
            if len(parsed_blocks) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} translations but received {len(parsed_blocks)}."
                )
            return [parsed_blocks[index].strip() for index in range(1, expected_count + 1)]

        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json") :].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        if "{" in cleaned and "}" in cleaned:
            cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

        payload = json.loads(cleaned)
        translations = payload.get("translations", [])
        if len(translations) != expected_count:
            raise ValueError(
                f"Expected {expected_count} translations but received {len(translations)}."
            )

        translations_sorted = sorted(translations, key=lambda item: item["id"])
        return [str(item.get("translated_text", "")).strip() for item in translations_sorted]

    def _should_fallback_to_single_item(self, exc: Exception) -> bool:
        message = str(exc)
        fallback_signals = (
            "Expected ",
            "Unterminated string",
            "Expecting ',' delimiter",
            "Expecting value",
            "Extra data",
        )
        return any(signal in message for signal in fallback_signals)
