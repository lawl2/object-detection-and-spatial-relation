from torchvision import transforms
import torch
import json
import os
import numpy as np
from PIL import Image
import cv2



class ClevrDataset(torch.utils.data.Dataset):
    """
    A class used to represent the dataset
    used for the model.
    It can be used to get datas both for 
    train and test. 
    The main difference between the two modes
    is that it gets data from different folders.
    Before training we remove a set of images in 
    order to obtain better results.

    Attributes
    ----------
    root_dir : str
        the path of the data root folder
    img_dir : str
        the path of the images folder (train or test)
    masks_dir : str
        the path of the masks folder (train or test)
    json_file : str
        the path of the json containing images info
    imgs : list of str
        a list of files where images are saved
    masks : list of str
        a list of files where masks are saved
    ign_imgs : list of str
        a list of filenames of images to remove


    Methods
    -------
    __getitem__(self, idx)
        get the image of index idx and the corresponding informations
        (bounding boxes, maks, label, etc)
    __len__(self)
        return the length of the dataset

    """
    def __init__(self, transforms, test: bool = False, test_size: int = -1):
        self.transforms = transforms
        self.root_dir = 'data'
        self.test = test

        if not test:
            self.img_dir = self.root_dir + '/train/images'
            self.masks_dir = self.root_dir + '/train/masks'
            self.ign_imgs = self.root_dir + "/train/rem_img/"

            self.imgs = list(sorted(os.listdir(self.img_dir)))
            self.imgs = [ele for ele in self.imgs if ele not in self.ign_imgs]

            self.masks = list(sorted(os.listdir(self.masks_dir)))

            with open(self.root_dir + "/CLEVR_train_bbox.json", 'r') as f:
                self.json_file = json.load(f)

        else:
            self.img_dir = self.root_dir + '/test/images'
            self.masks_dir = self.root_dir + '/test/masks'

            self.imgs = list(sorted(os.listdir(self.img_dir)))
            self.imgs.remove('.gitkeep')
            self.imgs = self.imgs[:test_size]

            self.masks = list(sorted(os.listdir(self.masks_dir)))
            self.masks = self.masks[:test_size]

            with open(self.root_dir + "/CLEVR_test_bbox.json", 'r') as f:
                self.json_file = json.load(f)


    def __getitem__(self, idx: int):
       
        # load images and masks
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        if self.test:
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("L").convert("RGB")

        json_info = [j for j in self.json_file if os.path.splitext(self.imgs[idx])[0] in j['filename']]
        num_objs = len(json_info)
        boxes = np.zeros((num_objs, 4))
        labels = np.zeros(num_objs)
        masks = np.zeros((num_objs, 320, 480))
        centers = []
        for i, f in enumerate(json_info):
            bbox = f['bbox']
            flat_bbox = torch.FloatTensor([x for xs in bbox for x in xs])
            boxes[i,:] = flat_bbox

            splitted = f['filename'].split("_")
            if self.test:
                labels[i] = int(splitted[3])
            else:
                labels[i] = ( int(splitted[3]) % 3 ) + 1
            m = cv2.imread(os.path.join(self.masks_dir,f['filename']), cv2.IMREAD_UNCHANGED)
            m[m > 0] = 1
            masks[i,:,:] = m
            splitted[5] = splitted[5].replace('.png', '')
            centers.append((int(splitted[5]), int(splitted[4])))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.ones((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["centers"] = centers

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self) -> int:
        return len(self.imgs)
    