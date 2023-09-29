import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision

from dataset import ClevrDataset
from thirdparty.engine import train_one_epoch
from thirdparty.utils import collate_fn
from utils import get_model, get_transform, remove_double_masks


def training(dataset, num_classes, num_epochs=10):

    # define data loader
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)
    if os.path.exists('data/model/weights_customj4_1_BCK_2022_11_07_00.torch'):
        model.load_state_dict(torch.load("data/model/weights_customj4_1_BCK_2022_11_07_00.torch"))
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9, 
            weight_decay=0.005
        )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.1   
            )

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=2)
        # update the learning rate
        lr_scheduler.step()
    
    torch.save(model.state_dict(), "weights_customj3_1.torch")



if __name__ == "__main__":

    # prepare dataset for training
    dataset = ClevrDataset(get_transform(train=True))

    remove_double_masks(dataset.imgs, dataset.json_file, dataset.ign_imgs, dataset.img_dir)
    
    # 3 classes + background
    num_classes = 4

    training(dataset, num_classes)
