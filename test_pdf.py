import os

import fitz  # PyMuPDF
import numpy as np
import cv2
from paddleocr import PaddleOCR


def _draw_ocr_boxes(img_bgr, ocr_lines, color=(0, 255, 0), thickness=2):
    """Draw PaddleOCR line boxes (4-point polygons) on a BGR image copy."""
    vis = img_bgr.copy()
    if not ocr_lines:
        return vis
    for line in ocr_lines:
        pts = np.array(line[0], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    return vis


def _bgr_images_to_pdf(images_bgr, out_pdf_path):
    """Write one PDF page per image (PNG streams; cv2 assumes BGR like imwrite)."""
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


def extract_text_from_pdf(pdf_path, language="en", annotated_pdf_path=None):
    """
    Extracts text from a PDF file using PaddleOCR.

    If annotated_pdf_path is set, saves a PDF where each source page is rendered
    at the same zoom as OCR and overlaid with detection bounding boxes.
    """
    print("Initializing PaddleOCR...")
    # Initialize PaddleOCR. 
    # use_angle_cls=True helps identify and correct rotated text.
    ocr = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)

    print(f"Opening PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return ""

    full_extracted_text = ""
    annotated_pages = []

    # Iterate through all the pages in the PDF
    for page_num in range(len(doc)):
        print(f"Processing Page {page_num + 1} of {len(doc)}...")
        page = doc.load_page(page_num)

        # Render page to an image (Pixmap)
        # Using a Matrix scales up the resolution (2x here) to improve OCR accuracy
        zoom_matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=zoom_matrix)

        # Convert the Pixmap into a NumPy array that OpenCV/PaddleOCR can read
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert colorspace depending on whether the PDF rendered an Alpha channel
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Run PaddleOCR on the image array
        result = ocr.ocr(img_array, cls=True)

        # Extract and append the text from the result list
        lines = result[0] if result and result[0] is not None else []
        page_text = ""
        for line in lines:
            # line format: [[bounding_box], (text, confidence_score)]
            text = line[1][0]
            page_text += text + "\n"

        if annotated_pdf_path is not None:
            annotated_pages.append(_draw_ocr_boxes(img_array, lines))

        full_extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n"

    doc.close()

    if annotated_pdf_path is not None:
        _bgr_images_to_pdf(annotated_pages, annotated_pdf_path)
        print(f"Wrote annotated PDF: {annotated_pdf_path}")

    return full_extracted_text

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Replace 'sample.pdf' with the path to your actual PDF file
    pdf_file_path = "docs/CamScanner 05-14-2026 15.15 (1).pdf"
    annotated_out = "output/CamScanner 05-14-2026 15.15 (1).pdf"

    extracted_text = extract_text_from_pdf(
        pdf_file_path, annotated_pdf_path=annotated_out
    )
    
    print("\n--- Final Extracted Text ---\n")
    print(extracted_text)
    
    # Optional: Save the extracted text to a .txt file
    # with open("output.txt", "w", encoding="utf-8") as f:
    #     f.write(extracted_text)