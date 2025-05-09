from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, Normalize, ColorJitter

CLASS_NAMES_VI = ["tức giận", "ghê tởm", "sợ hãi", "vui vẻ", "buồn bã", "ngạc nhiên", "bình thường"]

class EmotionDataSet(Dataset) :
    def __init__(self, image_dir, label_file, mode = 'train', transform=None):
        self.transform = transform
        self.samples = []
        self.image_dir = image_dir
        
        with open(label_file, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                image_name = parts[0]
                face_id = parts[1]
                
                top = int(parts[2])      # ymin
                left = int(parts[3])     # xmin
                right = int(parts[4])    # xmax
                bottom = int(parts[5])   # ymax
                
                xmin = left
                ymin = top
                xmax = right
                ymax = bottom
                box = [xmin, ymin, xmax, ymax]
                
                class_id = int(parts[7])
                
                self.samples.append({
                    'image_name': image_name,
                    'label': class_id,
                    'box': box
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.image_dir, sample['image_name'])
        image = Image.open(image_path)
        
        xmin, ymin, xmax, ymax = sample['box']
        face_crop = image.crop((xmin, ymin, xmax, ymax))
        
        if self.transform :
            face_crop = self.transform(face_crop)
            
        return face_crop, sample['label']
    
    def show_sample(self, index):
        face_crop, label = self.__getitem__(index)

        # Kiểm tra nếu face_crop là Tensor, chuyển sang PIL Image
        if isinstance(face_crop, torch.Tensor):
            face_crop = face_crop.permute(1, 2, 0).numpy()  # Chuyển từ C x H x W sang H x W x C
            face_crop = Image.fromarray(face_crop)  # Chuyển từ numpy array sang PIL Image
                
        plt.imshow(face_crop)
        plt.title(f"Label: {CLASS_NAMES_VI[label]}")
        plt.axis('off')
        plt.show()

        
# dataset = EmotionDataSet(
#     image_dir='ExpW/data/image/origin.7z/origin',
#     label_file='ExpW/data/label/label.lst',
#     transform=train_transform
# )

# for i in range(10):
#     # print(dataset.__getitem__(0))
#     dataset.show_sample(i)