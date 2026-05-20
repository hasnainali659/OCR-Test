import os

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from paddleocr import TextDetection
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

PDF_ZOOM = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------
# 1. Initialize Models (Load once, keep in memory)
# ---------------------------------------------------------

# PaddleOCR 3.x: use TextDetection for boxes only (no rec pipeline overhead)
print("Loading PaddleOCR Detector...")
det_model = TextDetection(model_name="PP-OCRv5_server_det")

# Initialize GLM-OCR (0.9B parameters easily fits on consumer GPUs)
print("Loading GLM-OCR Recognizer...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
rec_model = GlmOcrForConditionalGeneration.from_pretrained(
    "zai-org/GLM-OCR",
    dtype=torch.bfloat16,
).eval().to(device)


def _is_pdf(path):
    return os.path.splitext(path)[1].lower() == ".pdf"


def _is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def default_annotated_output_path(input_path, output_dir="output"):
    """Pick annotated output path based on input type (PDF or image)."""
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


def load_document_pages(input_path, pdf_zoom=PDF_ZOOM):
    """Load a document from an image or PDF path.

    Returns a list of (page_index, bgr_image) tuples. Single images use page_index 0.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if _is_pdf(input_path):
        try:
            doc = fitz.open(input_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not open PDF: {input_path}") from e

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

        if "poly" in line:
            pts = np.array(line["poly"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        text = (line.get("text") or "").strip()
        if not text:
            continue

        box_h = max(y_max - y_min, 1)
        scale = max(0.35, min(0.75, box_h / 36))
        thickness = max(1, int(round(scale * 2)))
        display_text = text if len(text) <= 60 else text[:57] + "..."

        (tw, th), baseline = cv2.getTextSize(display_text, font, scale, thickness)
        label_y = y_min - 6
        if label_y - th < 0:
            label_y = y_max + th + 6

        cv2.rectangle(
            annotated,
            (x_min, label_y - th - 4),
            (x_min + tw + 6, label_y + baseline + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            display_text,
            (x_min + 3, label_y),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(img_bgr, results, output_path):
    """Draw detection boxes and GLM-OCR text, then save to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, _draw_annotations(img_bgr, results))
    return output_path


def save_annotated_document(input_path, results, output_path):
    """Save annotated output as PDF for PDF inputs, otherwise as an image."""
    pages = load_document_pages(input_path)
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


def process_page(img_bgr):
    # dt_polys: list of quads [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
    detection_results = det_model.predict(img_bgr)[0]["dt_polys"]
    
    extracted_data = []

    # ---------------------------------------------------------
    # 3. Pipeline: Crop -> Recognize
    # ---------------------------------------------------------
    for box in detection_results:
        # Flatten the coordinates to find the min/max X and Y for a clean bounding box
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Crop the specific line of text from the original image using NumPy slicing
        cropped_img_cv = img_bgr[y_min:y_max, x_min:x_max]
        
        # Convert OpenCV BGR format to PIL RGB for the Hugging Face model
        cropped_pil_img = Image.fromarray(cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB))
        
        # ---------------------------------------------------------
        # 4. Prompt the Vision LLM
        # ---------------------------------------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_pil_img},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(rec_model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = rec_model.generate(**inputs, max_new_tokens=128)

        transcription = processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
        poly = np.asarray(box).reshape(-1, 2).tolist()
        extracted_data.append({
            "coordinates": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            "poly": poly,
            "text": transcription,
        })

    return extracted_data


def process_handwritten_document(input_path):
    """Run hybrid detection + recognition on an image or PDF file path."""
    pages = load_document_pages(input_path)
    all_results = []
    multi_page = len(pages) > 1

    for page_idx, img in pages:
        page_results = process_page(img)
        if multi_page:
            page_num = page_idx + 1
            print(f"Processing page {page_num} of {len(pages)}...")
            for line in page_results:
                line["page"] = page_num
        all_results.extend(page_results)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid OCR on a handwritten document (image or PDF)."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="docs/OneDrive_1_5-20-2026/Amir_Hafeez.pdf",
        help="Path to an image (.jpg, .png, ...) or .pdf file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Annotated output path (defaults to output/<name>_hybrid_annotated.<ext>)",
    )
    args = parser.parse_args()

    input_path = args.input_path
    results = process_handwritten_document(input_path)

    for line in results:
        page_label = f"Page {line['page']} " if "page" in line else ""
        print(f"{page_label}Box {line['coordinates']} -> Text: {line['text']}")

    out_path = args.output or default_annotated_output_path(input_path)
    save_annotated_document(input_path, results, out_path)
    print(f"Saved annotated output to {out_path}")