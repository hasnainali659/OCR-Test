import os

import cv2
import numpy as np
import torch
from PIL import Image
from paddleocr import TextDetection
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

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


def save_annotated_image(image_path, results, output_path):
    """Draw detection boxes and GLM-OCR text, then save to disk."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated = img.copy()
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path


def process_handwritten_document(image_path):
    # Load the high-res document image
    img = cv2.imread(image_path)
    
    # ---------------------------------------------------------
    # 2. Run Detection (Get Bounding Boxes)
    # ---------------------------------------------------------
    # dt_polys: list of quads [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
    detection_results = det_model.predict(image_path)[0]["dt_polys"]
    
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
        cropped_img_cv = img[y_min:y_max, x_min:x_max]
        
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


if __name__ == "__main__":
    image_path = "docs/OneDrive_1_5-20-2026/Amir_Hafeez.pdf"
    results = process_handwritten_document(image_path)

    for line in results:
        print(f"Box {line['coordinates']} -> Text: {line['text']}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("output", f"{stem}_hybrid_annotated.jpg")
    save_annotated_image(image_path, results, out_path)
    print(f"Saved annotated image to {out_path}")