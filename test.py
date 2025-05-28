from argparse import ArgumentParser
import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from emotion_dataset import EmotionDataSet
from model import load_resnet

def get_args():
    parser = ArgumentParser(description="CNN test")
    parser.add_argument("--image-path", "-ip", type=str, default="img-test/me.jpg", help="Image path to test")
    parser.add_argument("--image-size", "-is", type=int, default=224, help="Image size for model input")
    parser.add_argument("--checkpoint", "-chkpt", type=str, default="trained_models/best_cnn.pt", help="Model checkpoint path")
    args = parser.parse_args()
    return args

def show_image_with_prediction(image_path, prediction_text):
    image_display = cv2.imread(image_path)
    if image_display is None:
        print(f"Cannot read image at {image_path}")
        return
    image_display = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    plt.imshow(image_display)
    plt.title(f"Prediction: {prediction_text}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    categories = EmotionDataSet.get_categories()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet(num_classes=len(categories))
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Checkpoint not found!")
        exit(0)

    model.eval()
    image_test = cv2.imread(args.image_path)
    if image_test is None:
        print(f"Cannot read image from {args.image_path}")
        exit(1)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    image_test = cv2.resize(image_test, (args.image_size, args.image_size))
    image_test = np.transpose(image_test, (2, 0, 1)) / 255.0
    image_test = image_test[None, :, :, :]
    image_test = torch.from_numpy(image_test).float().to(device)

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(image_test)
        probs = softmax(output)
        max_idx = torch.argmax(probs, dim=1).item()

    predicted_class = categories[max_idx]
    print(f"Prediction: {predicted_class}")
    show_image_with_prediction(args.image_path, predicted_class)
