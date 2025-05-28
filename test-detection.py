import torch
import cv2
from torchvision import transforms
from PIL import Image
from model import load_resnet
import numpy as np
import os
import matplotlib.pyplot as plt

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

def predict_image_pil(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        score, predicted = torch.max(probs, 1)

    label = categories[predicted.item()]
    return label, score.item()

def crop_face_and_predict(input_img_path):
    img = cv2.imread(input_img_path)
    if img is None:
        print("Failed to load image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No face detected.")
        return

    fig, ax = plt.subplots(1)
    ax.imshow(img_rgb)
    
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = img[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        label, confidence = predict_image_pil(face_pil)
        if confidence < 0.80:
            continue
        print(f"Face {i+1}: {label} ({confidence:.2f})")

        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
        ax.text(x, y - 10, f"{label} ({confidence:.2f})", color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.title("Detected Faces with Emotion Predictions")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_path = "img-test/me.jpg"
    crop_face_and_predict(input_path)
