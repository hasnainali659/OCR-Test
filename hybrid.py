import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import TextDetection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GLM_OCR_PROMPT = "Text Recognition:"
OPENROUTER_SYSTEM_PROMPT = (
    "You are an OCR engine. Transcribe only the visible text. "
    "Never describe the image, never explain, never use markdown."
)
OPENROUTER_OCR_PROMPT = (
    "Transcribe the text in this image. "
    "Reply with ONLY the raw characters exactly as written. "
    "No descriptions, labels, markdown, quotes, or extra words."
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-vl-8b-instruct"
DEFAULT_OPENROUTER_MODEL_URDU = "qwen/qwen3-vl-32b-instruct"
QAARI_MODEL_ID = "oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct"
QAARI_OCR_PROMPT = (
    "Return only the Urdu text visible in this image, exactly as written in Nastaliq script. "
    "Do not add explanations."
)

URDU_OPENROUTER_SYSTEM_PROMPT = (
    "You are an Urdu OCR engine for Nastaliq (Perso-Arabic) script. "
    "Return only transcribed Urdu text. Never describe the image."
)
URDU_OPENROUTER_OCR_PROMPT = (
    "Transcribe the Urdu text in Nastaliq script exactly as written. "
    "Preserve Urdu letters, diacritics, and ligatures. "
    "Do not transliterate to English or Roman Urdu. Output only the Urdu text."
)
BILINGUAL_OPENROUTER_SYSTEM_PROMPT = (
    "You are a bilingual OCR engine for English and Urdu (Nastaliq). "
    "Return only transcribed text. Never describe the image."
)
BILINGUAL_OPENROUTER_OCR_PROMPT = (
    "Transcribe all visible text exactly as written. "
    "Keep English in Latin script and Urdu in Nastaliq Perso-Arabic script. "
    "Do not transliterate Urdu to Roman letters or wrong Arabic forms. "
    "Output only the raw text."
)
EN_OPENROUTER_SYSTEM_PROMPT = OPENROUTER_SYSTEM_PROMPT
EN_OPENROUTER_OCR_PROMPT = OPENROUTER_OCR_PROMPT

_NARRATION_PREFIXES = (
    r"^based on the image provided[,\s:;-]*",
    r"^here is the text recognition[:\s-]*",
    r"^the text in the image (?:is|reads|shows|says)[:\s-]*",
    r"^the image shows[:\s-]*",
    r"^this image (?:shows|contains|displays)[:\s-]*",
    r"^the visible text (?:is|reads|says)[:\s-]*",
    r"^text recognition[:\s-]*",
    r"^ocr result[:\s-]*",
    r"^transcription[:\s-]*",
    r"^output[:\s-]*",
)
_NARRATION_FIELD_PREFIXES = (
    r"^(?:a |the )?(?:form )?field labeled\s*",
    r"^(?:a |the )?(?:form )?label(?:ed)?\s*",
    r"^a portion of (?:a )?(?:form|document)(?:\s+or\s+document)?\s*(?:related to|about|for|containing)\s*",
)

_det_model = None
_glm_processor = None
_glm_model = None
_glm_device = None
_qaari_processor = None
_qaari_model = None
_qaari_device = None


@dataclass
class PipelineConfig:
    backend: str = "openrouter"
    mode: str = "line"
    lang: str = "bilingual"
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    det_model_name: str = "PP-OCRv5_mobile_det"
    pdf_zoom: float = 1.5
    batch_size: int = 16
    max_new_tokens: int = 64
    line_merge_ratio: float = 0.6
    max_horizontal_gap: int = 100
    fragment_gap: int = 35
    crop_padding: int = 6
    min_crop_height: int = 48
    image_max_side: int = 2048
    jpeg_quality: int = 95
    merge_fragments: bool = True


def sanitize_ocr_text(text):
    """Strip chatty model preambles and keep only transcribed content."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = cleaned.replace("??", "").strip()

    for _ in range(4):
        original = cleaned
        for pattern in _NARRATION_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
        for pattern in _NARRATION_FIELD_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == original:
            break

    for sep in ("reads:", "reads as:", "recognition:", "transcription:", "is:"):
        lower = cleaned.lower()
        if sep in lower:
            candidate = cleaned[lower.rfind(sep) + len(sep) :].strip()
            if candidate:
                cleaned = candidate

    cleaned = cleaned.strip("\"'` ")
    cleaned = re.sub(r"^[=\-–—:]+\s*", "", cleaned)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned).strip()
    return cleaned


def get_openrouter_prompts(config):
    if config.lang == "ur":
        return URDU_OPENROUTER_SYSTEM_PROMPT, URDU_OPENROUTER_OCR_PROMPT
    if config.lang == "en":
        return EN_OPENROUTER_SYSTEM_PROMPT, EN_OPENROUTER_OCR_PROMPT
    return BILINGUAL_OPENROUTER_SYSTEM_PROMPT, BILINGUAL_OPENROUTER_OCR_PROMPT


def load_dotenv(path=".env"):
    """Load KEY=VALUE pairs from .env without extra dependencies."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_openrouter_api_key():
    load_dotenv()
    for key in ("OPENROUTER_API_KEY", "OPENAI_ROUTER_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise RuntimeError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env "
        "(or OPENAI_ROUTER_KEY)."
    )


def _is_pdf(path):
    return os.path.splitext(path)[1].lower() == ".pdf"


def _is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def default_annotated_output_path(input_path, output_dir="output"):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if _is_pdf(input_path):
        ext = ".pdf"
    else:
        ext = os.path.splitext(input_path)[1].lower() or ".jpg"
        if ext not in IMAGE_EXTENSIONS:
            ext = ".jpg"
    return os.path.join(output_dir, f"{stem}_hybrid_annotated{ext}")


def _pixmap_to_bgr(pix):
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


def load_document_pages(input_path, pdf_zoom=1.5):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if _is_pdf(input_path):
        try:
            doc = fitz.open(input_path)
        except Exception as exc:
            raise FileNotFoundError(f"Could not open PDF: {input_path}") from exc

        pages = []
        zoom_matrix = fitz.Matrix(pdf_zoom, pdf_zoom)
        for page_num in range(len(doc)):
            pix = doc.load_page(page_num).get_pixmap(matrix=zoom_matrix)
            pages.append((page_num, _pixmap_to_bgr(pix)))
        doc.close()
        return pages

    if not _is_image(input_path):
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | {".pdf"}))
        raise ValueError(
            f"Unsupported input type: {input_path}. Supported extensions: {supported}"
        )

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    return [(0, img)]


def get_det_model(det_model_name):
    global _det_model
    if _det_model is None or getattr(_det_model, "_model_name", None) != det_model_name:
        print(f"Loading PaddleOCR Detector ({det_model_name})...")
        _det_model = TextDetection(model_name=det_model_name)
        _det_model._model_name = det_model_name
    return _det_model


def get_glm_model():
    global _glm_processor, _glm_model, _glm_device
    if _glm_model is not None:
        return _glm_processor, _glm_model, _glm_device

    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    print("Loading GLM-OCR Recognizer...")
    _glm_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if _glm_device == "cuda" else torch.float32
    _glm_processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
    _glm_model = GlmOcrForConditionalGeneration.from_pretrained(
        "zai-org/GLM-OCR",
        dtype=dtype,
    ).eval().to(_glm_device)
    print(f"GLM-OCR loaded on {_glm_device}.")
    return _glm_processor, _glm_model, _glm_device


def get_qaari_model():
    global _qaari_processor, _qaari_model, _qaari_device
    if _qaari_model is not None:
        return _qaari_processor, _qaari_model, _qaari_device

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    print(f"Loading Qaari Urdu OCR ({QAARI_MODEL_ID})...")
    _qaari_device = "cuda" if torch.cuda.is_available() else "cpu"
    _qaari_processor = AutoProcessor.from_pretrained(QAARI_MODEL_ID)
    _qaari_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QAARI_MODEL_ID,
        torch_dtype="auto",
        device_map="auto" if _qaari_device == "cuda" else None,
    ).eval()
    if _qaari_device != "cuda":
        _qaari_model = _qaari_model.to(_qaari_device)
    print(f"Qaari loaded on {_qaari_device}.")
    return _qaari_processor, _qaari_model, _qaari_device


def _box_bounds(box):
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return x_min, y_min, x_max, y_max


def _bounds_to_entry(box, text=""):
    x_min, y_min, x_max, y_max = _box_bounds(box)
    poly = np.asarray(box).reshape(-1, 2).tolist()
    return {
        "coordinates": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "poly": poly,
        "text": text,
    }


def _normalize_polys(polys):
    """Convert Paddle dt_polys (often a NumPy array) to a plain Python list."""
    if polys is None:
        return []
    if hasattr(polys, "tolist"):
        return polys.tolist()
    return list(polys)


def detect_boxes(img_bgr, det_model_name):
    det_model = get_det_model(det_model_name)
    result = det_model.predict(img_bgr)[0]
    return _normalize_polys(result.get("dt_polys"))


def merge_adjacent_fragments(boxes, config):
    """Join tiny adjacent detection boxes on the same row (common with Urdu Nastaliq)."""
    if not boxes:
        return []

    items = []
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_bounds(box)
        height = max(y_max - y_min, 1)
        items.append(
            {
                "box": box,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "y_center": (y_min + y_max) / 2.0,
                "height": height,
            }
        )

    items.sort(key=lambda item: (item["y_center"], item["x_min"]))
    merged_items = [items[0]]
    for item in items[1:]:
        prev = merged_items[-1]
        y_threshold = max(prev["height"], item["height"]) * config.line_merge_ratio
        x_gap = item["x_min"] - prev["x_max"]
        if (
            abs(item["y_center"] - prev["y_center"]) <= y_threshold
            and x_gap <= config.fragment_gap
        ):
            merged_items[-1] = {
                "box": [
                    [min(prev["x_min"], item["x_min"]), min(prev["y_min"], item["y_min"])],
                    [max(prev["x_max"], item["x_max"]), min(prev["y_min"], item["y_min"])],
                    [max(prev["x_max"], item["x_max"]), max(prev["y_max"], item["y_max"])],
                    [min(prev["x_min"], item["x_min"]), max(prev["y_max"], item["y_max"])],
                ],
                "x_min": min(prev["x_min"], item["x_min"]),
                "y_min": min(prev["y_min"], item["y_min"]),
                "x_max": max(prev["x_max"], item["x_max"]),
                "y_max": max(prev["y_max"], item["y_max"]),
                "y_center": (min(prev["y_min"], item["y_min"]) + max(prev["y_max"], item["y_max"])) / 2.0,
                "height": max(prev["height"], item["height"]),
            }
        else:
            merged_items.append(item)

    return [item["box"] for item in merged_items]


def prepare_detection_boxes(boxes, config):
    if config.merge_fragments and config.lang in ("ur", "bilingual"):
        boxes = merge_adjacent_fragments(boxes, config)
    return boxes


def merge_boxes_into_lines(boxes, merge_ratio=0.6, max_horizontal_gap=100):
    """Cluster detection boxes into text lines, splitting English/Urdu columns."""
    if not boxes:
        return []

    items = []
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_bounds(box)
        height = max(y_max - y_min, 1)
        items.append(
            {
                "box": box,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "y_center": (y_min + y_max) / 2.0,
                "height": height,
            }
        )

    items.sort(key=lambda item: (item["y_center"], item["x_min"]))
    row_groups = []
    for item in items:
        if not row_groups:
            row_groups.append([item])
            continue

        current = row_groups[-1]
        ref = current[-1]
        threshold = max(ref["height"], item["height"]) * merge_ratio
        if abs(item["y_center"] - ref["y_center"]) <= threshold:
            current.append(item)
        else:
            row_groups.append([item])

    merged = []
    for row_items in row_groups:
        row_items.sort(key=lambda item: item["x_min"])
        segments = [[row_items[0]]]
        for item in row_items[1:]:
            gap = item["x_min"] - segments[-1][-1]["x_max"]
            if gap > max_horizontal_gap:
                segments.append([item])
            else:
                segments[-1].append(item)

        for segment in segments:
            x_min = min(item["x_min"] for item in segment)
            y_min = min(item["y_min"] for item in segment)
            x_max = max(item["x_max"] for item in segment)
            y_max = max(item["y_max"] for item in segment)
            merged.append(
                {
                    "box": [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
    return merged


def crop_box(img_bgr, bounds, padding=0):
    h, w = img_bgr.shape[:2]
    x_min = max(0, bounds["x_min"] - padding)
    y_min = max(0, bounds["y_min"] - padding)
    x_max = min(w, bounds["x_max"] + padding)
    y_max = min(h, bounds["y_max"] + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = img_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def prepare_crop_for_ocr(pil_img, config):
    width, height = pil_img.size
    if height < config.min_crop_height:
        scale = config.min_crop_height / max(height, 1)
        pil_img = pil_img.resize(
            (max(1, int(width * scale)), config.min_crop_height),
            Image.Resampling.LANCZOS,
        )
    return pil_img


def pil_to_data_url(pil_img, config):
    """Resize large crops for API upload while preserving Urdu stroke detail."""
    img = pil_img.copy()
    img.thumbnail((config.image_max_side, config.image_max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=config.jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def recognize_openrouter(pil_images, config):
    api_key = get_openrouter_api_key()
    system_prompt, user_prompt = get_openrouter_prompts(config)
    texts = [""] * len(pil_images)

    def _call(index, pil_img):
        payload = {
            "model": config.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_data_url(pil_img, config)},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": config.max_new_tokens,
            "temperature": 0,
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/hasnainali659/OCR-Test",
                "X-Title": "paddle-ocr-test",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {body}")
        message = choices[0].get("message") or {}
        return index, sanitize_ocr_text(message.get("content") or "")

    workers = min(len(pil_images), config.batch_size, 8)
    if workers <= 1:
        for idx, pil_img in enumerate(pil_images):
            _, text = _call(idx, pil_img)
            texts[idx] = text
        return texts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_call, idx, pil_img)
            for idx, pil_img in enumerate(pil_images)
        ]
        for future in as_completed(futures):
            idx, text = future.result()
            texts[idx] = text
    return texts


def recognize_qaari_batch(pil_images, config):
    import torch

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Qaari backend requires qwen-vl-utils. Install with: pip install qwen-vl-utils"
        ) from exc

    processor, model, device = get_qaari_model()
    texts = []

    for pil_img in pil_images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": QAARI_OCR_PROMPT},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        texts.append(sanitize_ocr_text(decoded))

    return texts


def recognize_glm_batch(pil_images, config):
    import torch

    processor, model, device = get_glm_model()
    texts = []

    for start in range(0, len(pil_images), config.batch_size):
        batch_images = pil_images[start : start + config.batch_size]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": GLM_OCR_PROMPT},
                    ],
                }
            ]
            for pil_img in batch_images
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        for row in range(outputs.shape[0]):
            decoded = sanitize_ocr_text(
                processor.decode(
                    outputs[row][input_len:],
                    skip_special_tokens=True,
                )
            )
            texts.append(decoded)

    return texts


def recognize_crops(pil_images, config):
    if not pil_images:
        return []
    prepared = [prepare_crop_for_ocr(img, config) for img in pil_images]
    if config.backend == "openrouter":
        return recognize_openrouter(prepared, config)
    if config.backend == "qaari":
        return recognize_qaari_batch(prepared, config)
    return recognize_glm_batch(prepared, config)


def _crop_from_box(img_bgr, box, config):
    x_min, y_min, x_max, y_max = _box_bounds(box)
    return crop_box(
        img_bgr,
        {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        padding=config.crop_padding,
    )


def process_page_accurate(img_bgr, config, progress_label=""):
    boxes = prepare_detection_boxes(detect_boxes(img_bgr, config.det_model_name), config)
    extracted = []
    total = len(boxes)

    for index, box in enumerate(boxes, start=1):
        if progress_label:
            print(f"{progress_label} box {index}/{total} (accurate)...")
        pil_img = _crop_from_box(img_bgr, box, config)
        if pil_img is None:
            continue
        text = recognize_crops([pil_img], config)[0]
        extracted.append(_bounds_to_entry(box, text))

    return extracted


def process_page_line(img_bgr, config, progress_label=""):
    boxes = prepare_detection_boxes(detect_boxes(img_bgr, config.det_model_name), config)
    lines = merge_boxes_into_lines(
        boxes,
        config.line_merge_ratio,
        config.max_horizontal_gap,
    )
    if progress_label:
        print(f"{progress_label} {len(boxes)} boxes -> {len(lines)} lines.")

    pil_images = []
    line_meta = []
    for line in lines:
        pil_img = crop_box(img_bgr, line, padding=config.crop_padding)
        if pil_img is None:
            continue
        pil_images.append(pil_img)
        line_meta.append(line)

    if progress_label and pil_images:
        print(f"{progress_label} recognizing {len(pil_images)} lines...")

    texts = recognize_crops(pil_images, config)
    return [
        _bounds_to_entry(line["box"], text)
        for line, text in zip(line_meta, texts)
    ]


def process_page_page_mode(img_bgr, config, progress_label=""):
    h, w = img_bgr.shape[:2]
    if progress_label:
        print(f"{progress_label} full-page recognition...")

    page_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    page_text = recognize_crops([page_pil], config)[0]

    results = [
        {
            "coordinates": {"x_min": 0, "y_min": 0, "x_max": w, "y_max": h},
            "poly": [[0, 0], [w, 0], [w, h], [0, h]],
            "text": page_text,
            "full_page": True,
        }
    ]

    boxes = prepare_detection_boxes(detect_boxes(img_bgr, config.det_model_name), config)
    for box in boxes:
        entry = _bounds_to_entry(box, "")
        entry["detection_only"] = True
        results.append(entry)

    return results


def process_page(img_bgr, config, progress_label=""):
    if config.mode == "accurate":
        return process_page_accurate(img_bgr, config, progress_label)
    if config.mode == "page":
        return process_page_page_mode(img_bgr, config, progress_label)
    return process_page_line(img_bgr, config, progress_label)


def process_handwritten_document(input_path, config):
    pages = load_document_pages(input_path, pdf_zoom=config.pdf_zoom)
    all_results = []
    multi_page = len(pages) > 1

    print(
        f"Backend={config.backend}, mode={config.mode}, lang={config.lang}, "
        f"det={config.det_model_name}, zoom={config.pdf_zoom}"
    )

    for page_idx, img in pages:
        page_num = page_idx + 1
        label = f"Page {page_num}/{len(pages)}" if multi_page else "Page 1/1"
        page_results = process_page(img, config, progress_label=label)
        if multi_page:
            for line in page_results:
                line["page"] = page_num
        all_results.extend(page_results)

    return all_results


def _bgr_images_to_pdf(images_bgr, out_pdf_path):
    parent = os.path.dirname(os.path.abspath(out_pdf_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_doc = fitz.open()
    for img_bgr in images_bgr:
        h, w = img_bgr.shape[:2]
        ok, png_buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed while building annotated PDF")
        page = out_doc.new_page(width=w, height=h)
        page.insert_image(fitz.Rect(0, 0, w, h), stream=png_buf.tobytes())
    out_doc.save(out_pdf_path)
    out_doc.close()


def _draw_annotations(img_bgr, results):
    annotated = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for line in results:
        c = line["coordinates"]
        x_min, y_min, x_max, y_max = c["x_min"], c["y_min"], c["x_max"], c["y_max"]

        color = (255, 180, 0) if line.get("full_page") else (0, 255, 0)
        thickness = 3 if line.get("full_page") else 2

        if "poly" in line:
            pts = np.array(line["poly"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)

        text = (line.get("text") or "").strip()
        if not text:
            continue

        box_h = max(y_max - y_min, 1)
        scale = max(0.35, min(0.75, box_h / 36))
        if line.get("full_page"):
            scale = 0.55
        thickness_text = max(1, int(round(scale * 2)))
        display_text = text if len(text) <= 60 else text[:57] + "..."

        (tw, th), baseline = cv2.getTextSize(display_text, font, scale, thickness_text)
        label_y = y_min - 6
        if label_y - th < 0:
            label_y = y_max + th + 6

        cv2.rectangle(
            annotated,
            (x_min, label_y - th - 4),
            (x_min + tw + 6, label_y + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            display_text,
            (x_min + 3, label_y),
            font,
            scale,
            (0, 0, 0),
            thickness_text,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(img_bgr, results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, _draw_annotations(img_bgr, results))
    return output_path


def save_annotated_document(input_path, results, output_path, pdf_zoom=1.5):
    pages = load_document_pages(input_path, pdf_zoom=pdf_zoom)
    multi_page = len(pages) > 1

    if _is_pdf(input_path):
        annotated_pages = []
        for page_idx, img in pages:
            page_num = page_idx + 1
            page_results = [
                line for line in results if line.get("page", 1) == page_num
            ] if multi_page else results
            annotated_pages.append(_draw_annotations(img, page_results))
        _bgr_images_to_pdf(annotated_pages, output_path)
    else:
        save_annotated_image(pages[0][1], results, output_path)

    return output_path


def build_config_from_args(args):
    mode = args.mode
    lang = args.lang
    urdu_doc = lang in ("ur", "bilingual")

    if mode == "accurate":
        det_model = args.det_model or "PP-OCRv5_server_det"
        pdf_zoom = args.pdf_zoom if args.pdf_zoom is not None else (2.0 if urdu_doc else 2.0)
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (96 if urdu_doc else 128)
    elif mode == "page":
        det_model = args.det_model or "PP-OCRv5_mobile_det"
        pdf_zoom = args.pdf_zoom if args.pdf_zoom is not None else (2.0 if urdu_doc else 1.5)
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 512
    else:
        det_model = args.det_model or "PP-OCRv5_mobile_det"
        pdf_zoom = args.pdf_zoom if args.pdf_zoom is not None else (2.0 if urdu_doc else 1.5)
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (96 if urdu_doc else 64)

    model = args.model
    if (
        args.backend == "openrouter"
        and model == DEFAULT_OPENROUTER_MODEL
        and urdu_doc
    ):
        model = DEFAULT_OPENROUTER_MODEL_URDU
        print(f"Using Urdu-optimized OpenRouter model: {model}")

    backend = args.backend
    if backend == "glm":
        import torch

        if not torch.cuda.is_available():
            print(
                "Warning: CUDA not available. GLM-OCR on CPU is very slow; "
                "consider --backend openrouter or --backend qaari."
            )
    if backend == "qaari" and lang == "en":
        print("Warning: Qaari is Urdu-only. English text quality may be poor.")

    return PipelineConfig(
        backend=backend,
        mode=mode,
        lang=lang,
        openrouter_model=model,
        det_model_name=det_model,
        pdf_zoom=pdf_zoom,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        line_merge_ratio=args.line_merge_ratio,
        max_horizontal_gap=args.horizontal_gap,
        fragment_gap=args.fragment_gap,
        crop_padding=args.crop_padding,
        min_crop_height=args.min_crop_height,
        merge_fragments=not args.no_merge_fragments,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid OCR: PaddleOCR detection + GLM-OCR or OpenRouter/Qwen recognition.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="docs/OneDrive_1_5-20-2026/Sana Patel.pdf",
        help="Path to an image (.jpg, .png, ...) or .pdf file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Annotated output path (defaults to output/<name>_hybrid_annotated.<ext>)",
    )
    parser.add_argument(
        "--backend",
        choices=("glm", "openrouter", "qaari"),
        default="openrouter",
        help="Recognition backend (default: openrouter). qaari=local Urdu-specialized model.",
    )
    parser.add_argument(
        "--mode",
        choices=("page", "line", "accurate"),
        default="line",
        help="page=1 call/page, line=merged lines (recommended), accurate=1 call/box (slow)",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "ur", "bilingual"),
        default="bilingual",
        help="Language focus for prompts and preprocessing (default: bilingual)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_OPENROUTER_MODEL})",
    )
    parser.add_argument(
        "--det-model",
        default=None,
        help="Paddle detection model (default: mobile for line/page, server for accurate)",
    )
    parser.add_argument(
        "--pdf-zoom",
        type=float,
        default=None,
        help="PDF render scale (default: 2.0 for Urdu/bilingual, 1.5 otherwise)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for line-mode recognition (default: 16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max generated tokens per recognition call",
    )
    parser.add_argument(
        "--line-merge-ratio",
        type=float,
        default=0.6,
        help="Line clustering threshold as a fraction of box height (default: 0.6)",
    )
    parser.add_argument(
        "--horizontal-gap",
        type=int,
        default=100,
        help="Split merged lines when horizontal gap exceeds this many pixels (default: 100)",
    )
    parser.add_argument(
        "--fragment-gap",
        type=int,
        default=35,
        help="Merge adjacent same-row boxes when gap <= this many pixels (default: 35)",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=6,
        help="Padding around each crop in pixels (default: 6)",
    )
    parser.add_argument(
        "--min-crop-height",
        type=int,
        default=48,
        help="Upscale small crops to at least this height in pixels (default: 48)",
    )
    parser.add_argument(
        "--no-merge-fragments",
        action="store_true",
        help="Disable merging of tiny adjacent Urdu detection fragments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_args(args)

    started = time.perf_counter()
    input_path = args.input_path
    results = process_handwritten_document(input_path, config)
    elapsed = time.perf_counter() - started

    for line in results:
        if line.get("detection_only"):
            continue
        page_label = f"Page {line['page']} " if "page" in line else ""
        prefix = "[full page] " if line.get("full_page") else ""
        print(f"{page_label}{prefix}{line['coordinates']} -> {line['text']}")

    out_path = args.output or default_annotated_output_path(input_path)
    save_annotated_document(input_path, results, out_path, pdf_zoom=config.pdf_zoom)
    print(f"Saved annotated output to {out_path}")
    print(f"Done in {elapsed:.1f}s ({len(results)} result entries).")
