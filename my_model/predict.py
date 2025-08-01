import cv2
import numpy as np
import tensorflow as tf

# โหลดโมเดล
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# โหลดชื่อคลาส (ถ้ามี)
try:
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except:
    labels = ["Class 0", "Class 1", "Class 2"]

# เปิดกล้อง
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ ไม่สามารถถ่ายภาพจากกล้องได้")
    exit()

# เตรียมภาพ
img = cv2.resize(frame, (224, 224))  # ขนาดต้องตรงกับตอนเทรน
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# ส่งเข้าโมเดล
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# แสดงผล
predicted_index = int(np.argmax(output))
confidence = float(np.max(output))

print(f"✅ ทำนายว่า: {labels[predicted_index]} ({confidence:.2f})")
