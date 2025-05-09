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
 

def get_args():
    parser = ArgumentParser(description="Emotion Recognition CNN training")

    parser.add_argument("--image-dir", "-img-dir", type=str, default='ExpW/data/image/origin.7z/origin', help="Image of the dataset")
    parser.add_argument("--label-file", "-lbl", type=str, default='ExpW/data/label/label.lst', help="Label of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="Logging method")
    parser.add_argument("--trained-models", "-tr", type=str, default="trained_models", help="Model output dir")
    parser.add_argument("--checkpoint", "-chkpt", type=str, default="trained_models/best_model.pth", help="Path to save best model")

    return parser.parse_args()

def train_model():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    
    writer = SummaryWriter(args.logging)
    
    model = load_resnet()
    
    optimizer = optim.SGD([
        {'params': model.layer2.parameters(), 'lr': 1e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-3},
        {'params': model.fc.parameters(), 'lr': 1e-2}
    ], momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0

if __name__ == "__main__":
    train_model()