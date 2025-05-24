from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
from PIL import Image

CLASS_NAMES_VI = ["tức giận", "ghê tởm", "sợ hãi", "vui vẻ", "buồn bã", "ngạc nhiên", "bình thường"]

class EmotionDataSet(Dataset):
    def __init__(self, image_dir, label_file, mode='train', transform=None,
                 val_ratio=0.1, test_ratio=0.1, seed=42):
        self.transform = transform
        self.image_dir = image_dir
        self.categories = CLASS_NAMES_VI
        self.mode = mode
        self.samples = []

        all_samples = []
        with open(label_file, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                image_name = parts[0]
                box = [int(parts[3]), int(parts[2]), int(parts[4]), int(parts[5])]  # xmin, ymin, xmax, ymax
                class_id = int(parts[7])
                all_samples.append({
                    'image_name': image_name,
                    'label': class_id,
                    'box': box
                })

        trainval_samples, test_samples = train_test_split(
            all_samples,
            test_size=test_ratio,
            random_state=seed,
            stratify=[s['label'] for s in all_samples]
        )

        train_samples, val_samples = train_test_split(
            trainval_samples,
            test_size=val_ratio / (1 - test_ratio),
            random_state=seed,
            stratify=[s['label'] for s in trainval_samples]
        )

        if mode == 'train':
            self.samples = train_samples
        elif mode == 'val':
            self.samples = val_samples
        elif mode == 'test':
            self.samples = test_samples
        else:
            raise ValueError(f"Invalid mode {mode}, must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.image_dir, sample['image_name'])
        image = Image.open(image_path)
        xmin, ymin, xmax, ymax = sample['box']
        face_crop = image.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            face_crop = self.transform(face_crop)
        return face_crop, sample['label']
    
    @staticmethod
    def get_categories():
        return CLASS_NAMES_VI

# data = EmotionDataSet(
#     'ExpW/data/image/origin.7z/origin',
#     'ExpW/data/label/label.lst',
#     mode='train'
# )

# for i in range(10):
#     print(data.__getitem__(i)[1])