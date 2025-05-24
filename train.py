import torch
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, Normalize, ColorJitter, RandomHorizontalFlip, RandomRotation, RandomGrayscale
from emotion_dataset import EmotionDataSet
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from model import load_resnet
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix_and_save(writer, cm, class_names, epoch, save_dir='confusion_matrices'):
    os.makedirs(save_dir, exist_ok=True)

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm_normalized.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_normalized[i, j] > threshold else "black"
            plt.text(j, i, cm_normalized[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Lưu ảnh vào thư mục
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
    writer.add_figure('confusion_matrix', figure, epoch)
    plt.close()


def plot_confusion_matrix(writer, cm, class_names, epoch):

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = ArgumentParser(description="Emotion Recognition CNN training")

    parser.add_argument("--image-dir", "-img-dir", type=str, default='ExpW/data/image/origin.7z/origin', help="Image of the dataset")
    parser.add_argument("--label-file", "-lbl", type=str, default='ExpW/data/label/label.lst', help="Label of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="Logging method")
    parser.add_argument("--trained-models", "-tr", type=str, default="trained_models", help="Model output dir")
    parser.add_argument("--checkpoint", "-chkpt", type=str, default="trained_models\\last_cnn_1.pt", help="Path to save best model")

    return parser.parse_args()

def train_model():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("train with gpu")
    else:
        print("train with cpu")
    
    train_transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            
            RandomAffine(
                degrees=(-5, 5),
                translate=(0.1, 0.1),  # Dịch
                scale=(0.8, 1.2),  # zoom
            ),
            
            ColorJitter(brightness=0.1, contrast=0.3, saturation=0.4, hue=0.05), #màu
            
            RandomHorizontalFlip(), # lật ngang
                        
            RandomGrayscale(p=0.2), # ảnh xám
                        
            ToTensor(),
            
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
    )

    val_trainform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    train_dataset = EmotionDataSet(image_dir=args.image_dir, 
                                   label_file=args.label_file, mode= 'train', transform= train_transform)
    val_dataset = EmotionDataSet(image_dir=args.image_dir, 
                                   label_file=args.label_file, mode= 'val', transform= val_trainform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    
    writer = SummaryWriter(args.logging)
    
    model = load_resnet()
    model = model.to(device)
    
    # optimizer = optim.SGD([
    #     {'params': model.layer2.parameters(), 'lr': 1e-5},
    #     {'params': model.layer3.parameters(), 'lr': 1e-4},
    #     {'params': model.layer4.parameters(), 'lr': 1e-3},
    #     {'params': model.fc.parameters(), 'lr': 1e-2}
    # ], momentum=0.9, weight_decay=1e-4)

    # optimizer = optim.SGD([
    #     {'params': model.layer2.parameters(), 'lr': 1e-5},
    #     {'params': model.layer3.parameters(), 'lr': 1e-4},
    #     {'params': model.layer4.parameters(), 'lr': 1e-3},
    #     {'params': model.fc.parameters(), 'lr': 1e-3}
    # ], momentum=0.9, weight_decay=1e-4)

    optimizer = optim.SGD([
    {'params': model.layer2.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
], momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0
        
    for epoch in range(start_epoch, args.epochs):

        model.train()
        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            #forward
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("epoch {}/{} iteration {}/{} loss {:.3f}".format(epoch + 1, args.epochs, i + 1, len(train_loader), loss))
            writer.add_scalar("Train/Loss", loss, epoch * len(train_loader) + i)
            #backward
            optimizer.zero_grad() # refresh buffer
            loss.backward() # gradient
            optimizer.step()  # update parameter
        scheduler.step()
        
        model.eval()
        all_predictions = []
        all_labels = []
        progress_bar = tqdm(val_loader)
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            all_labels.extend(labels)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions,1)
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)
                progress_bar.set_description("val epoch {}/{} iteration {}/{} loss {:.3f}".format(epoch + 1, args.epochs, i + 1, len(val_loader), loss))
        # all_predictions = [prediction.item() for prediction in all_predictions]
        # all_labels = [label.item() for label in all_labels]
        
        # plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=val_dataset.categories, epoch=epoch)
        
        # accuracy =  accuracy_score(all_labels, all_predictions)
        
        # print("Epoch {}/{} accuracy {:.3f}".format(epoch + 1, args.epochs, accuracy))
        # writer.add_scalar("Val/Accuracy", accuracy, epoch)
        all_predictions = [prediction.item() for prediction in all_predictions]
        all_labels = [label.item() for label in all_labels]

        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix_and_save(writer, cm, class_names=val_dataset.categories, epoch=epoch)

        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}/{} Accuracy: {:.3f}".format(epoch + 1, args.epochs, accuracy))

        # In classification report
        report = classification_report(all_labels, all_predictions, target_names=val_dataset.categories, digits=4)
        print(report)

        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        checkpoint = {
            "best_acc": best_acc,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            best_acc = accuracy
            # torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))    
if __name__ == "__main__":
    train_model()