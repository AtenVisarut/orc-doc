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

# 2. กำหนดค่าเริ่มต้นและโหลดโมเดล
MODEL_PATH = "best.pt"

# ตรวจสอบว่ามีไฟล์โมเดลอยู่หรือไม่
if not os.path.exists(MODEL_PATH):
    print(f"ข้อผิดพลาด: ไม่พบไฟล์โมเดลที่เส้นทาง '{MODEL_PATH}'")
    print("กรุณาตรวจสอบให้แน่ใจว่าไฟล์ 'best.pt' อยู่ในไดเรกทอรีเดียวกันกับไฟล์ 'app.py'")
    # สามารถเพิ่มการจัดการข้อผิดพลาดที่เหมาะสม เช่น การหยุดแอปพลิเคชัน
    # exit()

# โหลดโมเดล YOLOv8 สำหรับการแบ่งส่วน (segmentation)
try:
    model = YOLO(MODEL_PATH)
    print("โมเดล YOLOv8 โหลดสำเร็จแล้ว")
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลดโมเดล YOLOv8: {e}")
    # สามารถเพิ่มการจัดการข้อผิดพลาดที่เหมาะสม
    # exit()

# โหลด EasyOCR Reader สำหรับภาษาไทยและอังกฤษ
try:
    reader = easyocr.Reader(['th', 'en'], gpu=False)
    print("EasyOCR Reader โหลดสำเร็จแล้ว")
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลด EasyOCR Reader: {e}")
    # สามารถเพิ่มการจัดการข้อผิดพลาดที่เหมาะสม
    # exit()

# 3. กำหนดคีย์เวิร์ดสำหรับการจัดกลุ่ม
keywords = {
    "ข้อมูลส่วนตัว": [
        "ชื่อ", "นามสกุล", "อายุ", "วันเกิด", "ที่อยู่", "เบอร์โทร", "โทรศัพท์", "อีเมล", "Email", "Contact", "Profile", 
        "เพศ", "สัญชาติ", "สถานะ", "เกิด", "ที่อยู่", "เบอร์ติดต่อ", "โทร.", "มือถือ",
        re.compile(r'\b(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),  # Email Regex
        re.compile(r'\b(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b')  # Phone number Regex
    ],
    "การศึกษา": [
        "การศึกษา", "การเรียน", "โรงเรียน", "มหาวิทยาลัย", "คณะ", "ปริญญา", "Degree", "Education", "University",
        "วิชา", "เกรด", "GPA", "รุ่น", "ปีที่เข้าศึกษา", "ปีที่จบ", "หลักสูตร", "สาขา", "major", "minor",
        "grade", "gpa", "graduation", "student", "course", "faculty", "department"
    ],
    "เกี่ยวกับฉัน": [
        "เกี่ยวกับฉัน", "สรุป", "แนะนำตัว", "About", "Summary", "Objective",
        "ประวัติ", "ประวัติส่วนตัว", "เป้าหมาย", "ความสนใจ", "ความถนัด", "เป้าหมายในชีวิต",
        "هدف", "هدفในอาชีพ", "interest", "hobby", "passion", "career objective", "career goal"
    ],
    "ทักษะ": [
        "ทักษะ", "ความสามารถ", "โปรแกรม", "ภาษา", "คอมพิวเตอร์", "ภาษาต่างประเทศ", "Software", "Skillset", "Skills",
        "ภาษาโปรแกรม", "framework", "tools", "เครื่องมือ", "ทักษะทางภาษา", "ทักษะคอมพิวเตอร์",
        "programming", "language", "framework", "tool", "technology", "skill", "ability", "competency",
        "certificate", "certification"
    ],
    "ประสบการณ์ทำงาน": [
        "ประสบการณ์", "ประสบการณ์ทำงาน", "ประวัติการทำงาน", "งาน", "ฝึกงาน", "Internship", "Project", "Experience", 
        "UX", "Designer", "ออกแบบ", "ตำแหน่ง", "บริษัท", "ระยะเวลา", "หน้าที่", "ความรับผิดชอบ", "โครงการ", "ผลงาน",
        "training", "position", "company", "duration", "responsibility", "duty", "achievement", "project",
        "work experience", "intern", "internship",
        re.compile(r'\b(20|19)\d{2}\b')  # Year Regex
    ],
    "กิจกรรม": [
        "กิจกรรม", "จิตอาสา", "อาสา", "volunteer", "activity", "club", "society", "organization",
        "ชมรม", "องค์กร", "โครงการสังคม", "social project"
    ],
    "รางวัล": [
        "รางวัล", "คะแนน", "แข่งขัน", "award", "prize", "competition", "contest", "scholarship",
        "ทุนการศึกษา", "achievement", "recognition", "honor"
    ],
    "ความสนใจ": [
        "ความสนใจ", "งานอดิเรก", "hobby", "interest", "pastime", "recreation"
    ],
    "อ้างอิง": [
        "อ้างอิง", "reference", "recommendation", "recommender", "ผู้แนะนำ"
    ]
}

# 4. กำหนดฟังก์ชันประมวลผลภาพ
def detect_and_crop_document(image_data, model, scale_factor=2.5):
    """
    ตรวจจับและตัดขอบเอกสารจากรูปภาพที่อัปโหลด

    Args:
        image_data (bytes): ข้อมูลไบต์ของรูปภาพ
        model (YOLO): โมเดล YOLO สำหรับการแบ่งส่วน
        scale_factor (float): ตัวคูณเพื่อปรับขนาดรูปภาพที่ตัดขอบแล้วให้ใหญ่ขึ้น เพื่อเพิ่มความแม่นยำของ OCR

    Returns:
        numpy.ndarray: รูปภาพที่ปรับปรุงแล้ว หรือ None หากไม่พบเอกสาร
    """
    # อ่านข้อมูลรูปภาพจากไบต์
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("ข้อผิดพลาด: ไม่สามารถถอดรหัสรูปภาพได้")
        return None
    
    h, w = img.shape[:2]

    # ทำนายการแบ่งส่วน (segmentation) ของเอกสาร
    results = model.predict(img, conf=0.5, verbose=False)
    r = results[0]
    
    # ตรวจสอบว่ามีหน้ากาก (mask) สำหรับเอกสารหรือไม่
    if not hasattr(r, 'masks') or r.masks is None or len(r.masks.data) == 0:
        print("ไม่พบหน้ากากเอกสารในภาพ")
        return None

    # แปลงหน้ากากเป็นรูปแบบที่ใช้ได้กับ OpenCV
    mask = r.masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ใช้หน้ากากเพื่อแยกเฉพาะส่วนของเอกสาร
    masked_document = cv2.bitwise_and(img, img, mask=mask)
    
    # ค้นหาโครงร่าง (contour) ของเอกสาร
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(largest_contour)
        
        # ตัดขอบรูปภาพตามกรอบที่พบ
        cropped = masked_document[y:y+h_c, x:x+w_c]
        
        # ปรับปรุงคุณภาพของรูปภาพ: แปลงเป็นขาว-ดำและปรับความสว่าง/ความคมชัด
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # ปรับขนาดรูปภาพให้ใหญ่ขึ้นเพื่อเพิ่มความแม่นยำของ OCR
        new_w = int(w_c * scale_factor)
        new_h = int(h_c * scale_factor)
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    return None

# 5. ฟังก์ชันประมวลผล OCR
def ocr_easyocr_with_boxes(image, keywords):
    """
    ทำการ OCR บนรูปภาพและวาดกรอบข้อความพร้อมจำแนกตามคีย์เวิร์ด

    Args:
        image (numpy.ndarray): รูปภาพที่ผ่านการปรับปรุงแล้ว
        keywords (dict): พจนานุกรมของคีย์เวิร์ดสำหรับจำแนกประเภท

    Returns:
        tuple: (รูปภาพที่มีกรอบข้อความ, พจนานุกรมของข้อความที่จำแนกแล้ว)
    """
    # ทำ OCR บนรูปภาพที่เข้ามา
    result = reader.readtext(image)
    topics = {k: [] for k in keywords}
    
    # แปลงรูปภาพเป็น BGR เพื่อวาดกรอบ
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # สร้างสีสุ่มสำหรับแต่ละประเภทเพื่อความสวยงาม
    color_map = {k: tuple([random.randint(0, 255) for _ in range(3)]) for k in keywords}

    for (bbox, text, prob) in result:
        assigned = False
        text_lower = text.lower()
        
        # ตรวจสอบคีย์เวิร์ดเพื่อจำแนกประเภทข้อความ
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
        
        # หากไม่สามารถจำแนกประเภทได้ ให้ใช้สีเทา
        if not assigned:
            color = (150, 150, 150)

        # วาดกรอบสี่เหลี่ยมรอบข้อความ
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)
        
        # วาดข้อความกำกับ
        cv2.putText(image_bgr, text, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return image_bgr, topics

# 6. ฟังก์ชันประมวลผลหลัง OCR
def postprocess_ocr_results(topics):
    """
    ประมวลผลผลลัพธ์หลัง OCR เพื่อเพิ่มความแม่นยำ
    """
    processed_topics = {}
    
    for category, texts in topics.items():
        unique_texts = []
        seen = set()
        
        for text in texts:
            # ลบช่องว่างและอักขระพิเศษ
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # ตรวจสอบว่าเป็นข้อความที่ซ้ำกัน
            if cleaned_text and cleaned_text not in seen:
                seen.add(cleaned_text)
                unique_texts.append(cleaned_text)
        
        processed_topics[category] = unique_texts
    
    return processed_topics

# 7. กำหนด Flask app และ routes
app = Flask(__name__)

@app.route("/")
def home():
    """Route สำหรับหน้าหลักที่แสดงไฟล์ HTML"""
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    """Route สำหรับรับรูปภาพและประมวลผล"""
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "ไม่พบไฟล์รูปภาพที่อัปโหลด"}), 400

    file = request.files["image"]
    image_data = file.read()

    try:
        # ตรวจจับและตัดขอบเอกสาร
        cropped_doc = detect_and_crop_document(image_data, model, scale_factor=2.5)

        if cropped_doc is None:
            return jsonify({"error": "ไม่พบเอกสารในภาพ โปรดลองรูปภาพอื่น"}), 404
        
        # ทำ OCR และวาดกรอบ
        highlighted_img, topics = ocr_easyocr_with_boxes(cropped_doc, keywords)
        
        # ประมวลผลผลลัพธ์หลัง OCR
        processed_topics = postprocess_ocr_results(topics)

        # แปลงรูปภาพผลลัพธ์เป็น Base64
        _, img_encoded = cv2.imencode(".png", highlighted_img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        # สร้างข้อความสำหรับแสดงผล
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

# 8. เริ่มการทำงานของแอปพลิเคชัน
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))