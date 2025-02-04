import cv2
import torch
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Загрузка предобученной модели YOLO
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_logo(image_path, reference_logo_path):
    # Загрузка изображения и эталонного логотипа
    image = cv2.imread(image_path)
    ref_logo = cv2.imread(reference_logo_path, cv2.IMREAD_GRAYSCALE)
    
    # Обнаружение объектов на изображении с помощью YOLOv5
    results = model_yolo(image)
    detections = results.xyxy[0]

    logo_found = False 

    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        cropped_logo = image[y1:y2, x1:x2]
        
        # Сравнение с эталонным логотипом (метод SIFT)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(cropped_logo, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(ref_logo, None)

        if des1 is None or des2 is None:
            continue  # Пропускаем итерацию, если дескрипторы не найдены
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Проверяем длину matches и фильтруем совпадения
        good_matches = []
        for match in matches:
            if len(match) == 2:  # Убедимся, что есть пара (m, n)
                m, n = match
                if m.distance < 0.75 * n.distance:  # Фильтрация по расстоянию косинусному
                    good_matches.append(m)

        if len(good_matches) > 10:  # Порог совпадений
            logo_found = True  # Логотип найден
            print(f"Логотип совпадает с эталоном! Координаты: ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2_imshow(image)


detect_logo(
    "/content/flipped_horizontal.jpg",
    "/content/pre-sekret-logotipa-starbucks-kotoryj-vy-veroyatno-nikogda-ne-zamechali.jpg"
)
