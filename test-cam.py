import torch
import cv2
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from model import load_resnet
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_checkpoint = 'trained_models/best_cnn.pt'
if not os.path.exists(model_checkpoint):
    print("Model checkpoint not found!")
    exit()

model = load_resnet(num_classes=7)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model'])
model.to(device)
model.eval()

categories = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def predict_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        score, predicted = torch.max(probs, 1)

    label = categories[predicted.item()]
    return label, score.item()

def draw_text_vietnamese(image, text, position, font_path="arial.ttf", font_size=24, color=(255, 0, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Cannot read frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        predicted_label, score = predict_image(face_pil)
        label_text = f"{predicted_label} ({score:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        frame = draw_text_vietnamese(frame, label_text, (x, y-30), font_size=20)
    try:
        cv2.imshow('Face Detection and Prediction', frame)
    except cv2.error as e:
        print(f"OpenCV imshow error: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
