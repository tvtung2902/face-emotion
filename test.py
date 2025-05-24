from argparse import ArgumentParser
import torch
from torch import nn
import cv2
import numpy as np
from emotion_dataset import EmotionDataSet
from model import load_resnet

def get_args():
    parser = ArgumentParser(description="CNN test")
    parser.add_argument("--image-path", "-ip", type=str, default="img.png", help="Image ")
    parser.add_argument("--image-size", "-is", type=int, default=224, help="Image size")
    parser.add_argument("--checkpoint", "-chkpt", type=str, default="trained_models/best_cnn.pt", help="checkpoint")
    args = parser.parse_args()
    return args

def show_image_with_prediction(image_path, prediction_text):
    image_display = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_display, f'Prediction: {prediction_text}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    window_name = 'Prediction Result'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen = np.array(cv2.getWindowImageRect(window_name))
    screen_width, screen_height = screen[2], screen[3]
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    img_height, img_width = image_display.shape[:2]

    x = (screen_width - img_width) // 2
    y = (screen_height - img_height) // 2

    cv2.moveWindow(window_name, x, y)

    cv2.imshow(window_name, image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    categories = EmotionDataSet.get_categories()
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('run with gpu')
    else:
        device = torch.device("cpu")

    model = load_resnet()
    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
    else:
        print('no checkpoint found!')
        exit(0)

    model.eval()
    image_test = cv2.imread(args.image_path)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    image_test = cv2.resize(image_test, (args.image_size, args.image_size))
    image_test = np.transpose(image_test, (2, 0, 1))/255.0
    image_test = image_test[None, :, :, :]
    image_test = torch.from_numpy(image_test).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image_test)
        print(output)
        print('----------------')
        print(softmax(output))

    max_idx = torch.argmax(softmax(output))
    predicted_class = categories[max_idx]
    print("test img is: ", predicted_class)

    show_image_with_prediction(args.image_path, predicted_class)