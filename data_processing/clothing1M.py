from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image

class clothing_dataset(Dataset): 
    def __init__(self, transform, mode): 
        self.train_imgs = []
        self.test_imgs = []
        self.train_labels = {}
        self.test_labels = {}
        self.transform = transform
        self.mode = mode
        with open('autodl-nas/MyClothing1M/data/annotations/noisy_train_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = 'autodl-nas/MyClothing1M/data/Clothing1M/'+l[7:]
            self.train_imgs.append(img_path)
            
        with open('autodl-nas/MyClothing1M/data/annotations/clean_train_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = 'autodl-nas/MyClothing1M/data/Clothing1M/'+l[7:]
            self.train_imgs.append(img_path)

        with open('autodl-nas/MyClothing1M/data/annotations/clean_test_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = 'autodl-nas/MyClothing1M/data/Clothing1M/'+l[7:]
            self.test_imgs.append(img_path)
            
        with open('autodl-nas/MyClothing1M/data/annotations/noisy_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = 'autodl-nas/MyClothing1M/data/Clothing1M/'+entry[0][7:]
            self.train_labels[img_path] = int(entry[1])

        with open('autodl-nas/MyClothing1M/data/annotations/clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = 'autodl-nas/MyClothing1M/data/Clothing1M/'+entry[0][7:]
            self.test_labels[img_path] = int(entry[1])
            self.train_labels[img_path] = int(entry[1])
        random.Random(42).shuffle(self.train_imgs)
        
        '''from collections import Counter
        min_samples = 18500
        for i in range(14):
            num_samples = Counter(self.train_labels.values())[i]
            if num_samples < min_samples:
                min_samples = num_samples
        
        class_counts = np.zeros(14)
        train_imgs2 = []
        train_labels2 = {}
        for img in self.train_imgs:
            label = self.train_labels[img]
            if class_counts[label] < min_samples:
                train_imgs2.append(img)
                train_labels2[img] = label
                class_counts[label] += 1
        self.train_imgs = train_imgs2
        self.train_labels = train_labels2'''
            
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]           
        image = Image.open(img_path).convert('RGB')    
        img = self.transform(image)
        return img, target
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)   
        
class clothing_dataloader():  
    def __init__(self, batch_size, num_workers, shuffle):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
   
    def run(self):
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])

        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])    
                
        train_dataset = clothing_dataset(transform=self.transform_train, mode='train')
        test_dataset = clothing_dataset(transform=self.transform_test, mode='test')
        print(len(train_dataset), len(test_dataset))
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers)              
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)                   
        return train_loader, test_loader