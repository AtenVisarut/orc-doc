# -*- coding: utf-8 -*-

# 1. Import libraries
import os
import io
import cv2
import numpy as np
import base64
import re
import random
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import easyocr

# 2. กำหนดค่าเริ่มต้นและโหลดโมเดล YOLO
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"ข้อผิดพลาด: ไม่พบไฟล์โมเดลที่เส้นทาง '{MODEL_PATH}'")

try:
    model = YOLO(MODEL_PATH)
    print("โมเดล YOLOv8 โหลดสำเร็จแล้ว")
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลดโมเดล YOLOv8: {e}")

# EasyOCR ใช้ lazy load เพื่อลด RAM
reader = None

def get_reader():
    global reader
    if reader is None:
        # โหลดเฉพาะภาษาไทย (ถ้าต้องการภาษาอังกฤษให้แก้เป็น ['th','en'])
        reader = easyocr.Reader(['th'], gpu=False)
        print("EasyOCR Reader โหลดสำเร็จแล้ว")
    return reader

# 3. กำหนดคีย์เวิร์ดสำหรับการจัดกลุ่ม
keywords = {
    "ข้อมูลส่วนตัว": [
        "ชื่อ", "นามสกุล", "อายุ", "วันเกิด", "ที่อยู่", "เบอร์โทร", "โทรศัพท์", "อีเมล", "Email", "Contact", "Profile",
        "เพศ", "สัญชาติ", "สถานะ", "เกิด", "ที่อยู่", "เบอร์ติดต่อ", "โทร.", "มือถือ",
        re.compile(r'\b(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),
        re.compile(r'\b(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b')
    ],
    "การศึกษา": ["การศึกษา", "มหาวิทยาลัย", "คณะ", "ปริญญา", "Education", "University", "GPA"],
    "เกี่ยวกับฉัน": ["เกี่ยวกับฉัน", "สรุป", "แนะนำตัว", "About", "Objective"],
    "ทักษะ": ["ทักษะ", "ความสามารถ", "Skill", "language", "framework"],
    "ประสบการณ์ทำงาน": ["ประสบการณ์", "งาน", "Internship", "Experience", "Project"],
    "กิจกรรม": ["กิจกรรม", "อาสา", "volunteer", "activity"],
    "รางวัล": ["รางวัล", "award", "competition"],
    "ความสนใจ": ["ความสนใจ", "hobby", "interest"],
    "อ้างอิง": ["อ้างอิง", "reference"]
}

# 4. ฟังก์ชันตรวจจับและครอปเอกสาร
def detect_and_crop_document(image_data, model, scale_factor=1.5):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("ข้อผิดพลาด: ไม่สามารถถอดรหัสรูปภาพได้")
        return None
    
    h, w = img.shape[:2]
    results = model.predict(img, conf=0.5, verbose=False)
    r = results[0]
    
    if not hasattr(r, 'masks') or r.masks is None or len(r.masks.data) == 0:
        print("ไม่พบหน้ากากเอกสารในภาพ")
        return None

    mask = r.masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    masked_document = cv2.bitwise_and(img, img, mask=mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(largest_contour)
        cropped = masked_document[y:y+h_c, x:x+w_c]
        
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        new_w = int(w_c * scale_factor)
        new_h = int(h_c * scale_factor)
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    return None

# 5. OCR + วาดกรอบ
def ocr_easyocr_with_boxes(image, keywords):
    result = get_reader().readtext(image)
    topics = {k: [] for k in keywords}
    
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    color_map = {k: tuple([random.randint(0, 255) for _ in range(3)]) for k in keywords}

    for (bbox, text, prob) in result:
        assigned = False
        text_lower = text.lower()
        
        for k, kw_list in keywords.items():
            for kw in kw_list:
                if isinstance(kw, str) and kw.lower() in text_lower:
                    topics[k].append(text)
                    assigned = True
                    color = color_map[k]
                    break
                elif isinstance(kw, re.Pattern) and kw.search(text):
                    topics[k].append(text)
                    assigned = True
                    color = color_map[k]
                    break
            if assigned:
                break
        
        if not assigned:
            color = (150, 150, 150)

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(image_bgr, text, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return image_bgr, topics

# 6. Postprocess OCR
def postprocess_ocr_results(topics):
    processed_topics = {}
    for category, texts in topics.items():
        unique_texts = []
        seen = set()
        for text in texts:
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            if cleaned_text and cleaned_text not in seen:
                seen.add(cleaned_text)
                unique_texts.append(cleaned_text)
        processed_topics[category] = unique_texts
    return processed_topics

# 7. Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "ไม่พบไฟล์รูปภาพที่อัปโหลด"}), 400

    file = request.files["image"]
    image_data = file.read()

    try:
        cropped_doc = detect_and_crop_document(image_data, model, scale_factor=1.5)
        if cropped_doc is None:
            return jsonify({"error": "ไม่พบเอกสารในภาพ โปรดลองรูปภาพอื่น"}), 404
        
        highlighted_img, topics = ocr_easyocr_with_boxes(cropped_doc, keywords)
        processed_topics = postprocess_ocr_results(topics)

        _, img_encoded = cv2.imencode(".png", highlighted_img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        classified_text = ""
        for k, paras in processed_topics.items():
            if paras:
                classified_text += f"\n[{k}]\n"
                for p in paras:
                    classified_text += f"- {p}\n"

        return jsonify({
            "status": "success",
            "topics": classified_text,
            "highlighted_image": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"มีข้อผิดพลาดในการประมวลผล: {str(e)}"}), 500

# 8. Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
