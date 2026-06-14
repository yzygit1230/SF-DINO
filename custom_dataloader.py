"""
author:JIN
"""
from torchvision.datasets.vision import VisionDataset
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import accimage
import os
import os.path
import warnings
warnings.filterwarnings('ignore')

from torchvision import transforms

class Custom_split_dataloader(VisionDataset):
    def __init__(self, root, test_size=None,val_size=None,train=True,test=True,val=True,transform=None,
                 ):
        self.root=root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(self.root, class_to_idx)
        self.loader = self.default_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.train=train
        self.test=test
        self.val=val
        
        self.test_size=test_size
        self.val_size=val_size
        [self.samples_train_index, self.samples_test_index]= list(StratifiedShuffleSplit(
                                                                n_splits=1, test_size= self.test_size,random_state=42
                                                                ).split(self.samples, self.targets))[0]
        self.samples_trains = [self.samples[i] for i in self.samples_train_index]
        self.samples_test = [self.samples[i] for i in self.samples_test_index]
       
        self.samples_trains_targets = [s[1] for s in self.samples_trains]
        [self.samples_train_index, self.samples_val_index]= list(StratifiedShuffleSplit(
                                                            n_splits=1, test_size= self.val_size,random_state=42
                                                             ).split(self.samples_trains,  self.samples_trains_targets))[0]
       
        self.samples_train = [self.samples_trains[i] for i in self.samples_train_index]
        self.samples_val = [self.samples_trains[i] for i in self.samples_val_index]
        self.patch_dir = root
        self.transform = transform

        self.strong_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
            transforms.ToTensor()
        ])

        if self.train==True:
           self.imgs = self.samples_train      
           
        if self.test==True:
           self.imgs = self.samples_test
           
        if self.val==True:
           self.imgs = self.samples_val
       
        self.samples=self.imgs
    def pil_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    
    def accimage_loader(self,path):
        try:
            return accimage.Image(path)
        except IOError:
            
            return self.pil_loader(path)
    
    
    def default_loader(self,path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)
        
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _make_dataset(self, directory, class_to_idx):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
        return instances   
 
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
    










