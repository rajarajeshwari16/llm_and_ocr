import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import google.auth
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions


LOGGER = logging.getLogger("pdf_translate.translate")


def chunk_segments_for_translation(segments: Sequence, batch_size: int) -> Iterable[Sequence]:
    for index in range(0, len(segments), batch_size):
        yield segments[index : index + batch_size]


class VertexTranslator:
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
        lang_name: str = "Kannada",
    ) -> list:
        from pathlib import Path as _Path
        from google.genai import types as genai_types

        image_bytes = _Path(image_path).read_bytes()
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

        prompt = (
            f"You are analyzing a scanned {lang_name} legal/land document image.\n"
            "Extract every visible text block and translate each one to English.\n"
            "Return ONLY a valid JSON array. No markdown, no explanation, nothing else.\n"
            "Format:\n"
            '[\n'
            '  {"translated_text": "English text here", "x_pct": 0.05, "y_pct": 0.02, "w_pct": 0.9, "h_pct": 0.04},\n'
            '  ...\n'
            ']\n'
            "Rules:\n"
            "- x_pct, y_pct = top-left corner as fraction of image width/height (0.0 to 1.0)\n"
            "- w_pct, h_pct = width/height as fraction of image dimensions (0.0 to 1.0)\n"
            "- Preserve all names, numbers, survey numbers, dates, and legal terms exactly\n"
            "- Include every text block including table headers and cells\n"
            "- Do NOT add explanations or commentary\n"
            "- Output ONLY the JSON array"
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
