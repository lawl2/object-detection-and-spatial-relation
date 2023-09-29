import json
import os
import random
from fileinput import filename

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import thirdparty.transforms as T
from preprocessing import generate_mask

list_class = {'1': 'sphere',
              '2': 'cube',
              '3': 'cylinder'}

colors = ['gray', 'red', 'blue', 'brown', 'yellow', 'green', 'purple', 'cyan']


def get_model(num_classes):
    """
    Create and configure a MaskrRCNN predictor.
    The network has ResNet50 as backbone with pretrained weights,
    and use 128 hidden layers for the rest of the architecture.

    @type   num_classes : int
    @param  num_classes : number of classes 

    @rtype  torchvision.models.detection.mask_rcnn.MaskRCNN
    @returns  model : the model to be used
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 32
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    """
    Apply necessary transformation to the intput

    @type   train : bool
    @param  train : indicates if in training or testing mode
    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.6))
    return T.Compose(transforms)


def remove_double_masks(imgs, json_file, rem_img, img_dir):
    """
    Used to remove images that has generated a
    single mask for two or more objects close
    to each other and of the same color and material.
    This is done in order to have better results 
    on training

    @type   root_dir : str
    @param  root_dir : location of the data root dir
    
    @type   imgs : list 
    @param  imgs : list of filenames of images

    @type   json_file : str
    @param  json_file : location of json file with bounding box
    """
    dup_mask = []
    # iterate over images 
    for img in imgs:
        # list of each mask and corresponding bb of the same image
        mask_info = [j for j in json_file if os.path.splitext(img)[0] in j['filename']]
        # iterate over each object
        for f in mask_info:
            i = 0
            split = f['filename'].split("_")
            # {split[0]}{split[1]}{split[2]}.png is the name of the image we want to remove
            filename = f"{split[0]}_{split[1]}_{split[2]}.png"
            if filename in dup_mask:
                continue
            for j in mask_info:
                if f['bbox'] == j['bbox'] or f['bbox'] == [[0, 0], [0, 0]] or j['bbox'] == [[0, 0], [0, 0]] :
                    i += 1
                if i >= 2:
                    dup_mask.append(filename)
    dup_mask = list(dict.fromkeys(dup_mask))        
    for f in dup_mask:            
        os.replace(img_dir + "/" + f, rem_img + f)




def random_color_masks(image):
    """
    Generate a mask over an image
    """
    colors = [[255, 255, 255]]
    # I will copy a list of colors here
    #colors = [
    #    [0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],
    #    [255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],
    #    [70, 150, 250],[50, 190, 190]
    #]
    colors = [
        [255, 255, 255], [255, 255, 255], [255, 255, 255],
        [255, 255, 255], [255, 255, 255], [255, 255, 255], 
        [255, 255, 255], [255, 255, 255], [255, 255, 255],
        [255, 255, 255],  [255, 255, 255]
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, len(colors))]
    colored_mask = np.stack([r,g,b], axis=2)

    return colored_mask


def spatialPosition(info):
    """
    Calculates the number of object at right, left,
    above, below of the current object.
    Parameter info is updated with this informations

    @type   info: dict
    @param  info: dictionary with prediction informations
    """
    x = {x: x[0] for x in info['predict'].keys()}
    y = {y: y[1] for y in info['predict'].keys()}
    x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    y = {k: v for k, v in sorted(y.items(), key=lambda item: item[1])}

    left = []
    right = list(x)

    top = []
    down = list(y)
    
    for sx in x:
        curr = right.pop(0)
        cl = len([x for x in left if not x[0] == curr[0]])
        cr = len([x for x in right if not x[0] == curr[0]])
        info['predict'][sx]['lr_info'] = f"Left: {cl} Right: {cr}"
        left.append(curr)


    for sy in y:
        curr = down.pop(0)
        ct = len([x for x in top if not x[1] == curr[1]])
        cd = len([x for x in down if not x[1] == curr[1]])
        info['predict'][sy]['td_info'] = f"Above: {ct} Below: {cd}"
        top.append(curr)



def view(info):
    """
    Display predicions to video.

    @type   info: dict
    @param  info: dictionary with predictions data
    """

    spatialPosition(info)
    image = info['image'].copy()
    for i in info['predict']:

        image = cv2.rectangle(image, info['predict'][i]['start_point'], info['predict'][i]['end_point'], (255, 0, 0), 2)
        image = cv2.addWeighted(image, 1, random_color_masks(info['predict'][i]['mask']), 0.3, 0)
        label  = info['predict'][i]['label']
        image = cv2.putText(img=image, text=label, org=info['predict'][i]['start_point'], fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0, 255, 0),thickness=1)

        lr = info['image'].copy()
        cv2.line(lr, (int(info['predict'][i]['center'][0]), 0), (int(info['predict'][i]['center'][0]), 320), (0, 0, 255), 2)
        lr = cv2.addWeighted(lr, 1, random_color_masks(info['predict'][i]['mask']), 0.3, 0)
        lr = cv2.putText(img=lr, text=info['predict'][i]['lr_info'], org=info['predict'][i]['center'], fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(255, 255, 255),thickness=1)
        lr = cv2.putText(img=lr, text=f"{label} {info['predict'][i]['center']}", org=(20 ,10), fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=(255, 255, 255),thickness=1)

        td = info['image'].copy()
        cv2.line(td, (0, int(info['predict'][i]['center'][1])), (480, int(info['predict'][i]['center'][1])), (0, 0, 255), 2)
        td = cv2.addWeighted(td, 1, random_color_masks(info['predict'][i]['mask']), 0.3, 0)
        td = cv2.putText(img=td, text=info['predict'][i]['td_info'], org=info['predict'][i]['center'], fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(255, 255, 255),thickness=1)
        td = cv2.putText(img=td, text=f"{label} {info['predict'][i]['center']}", org=(20 ,10), fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=(255, 255, 255),thickness=1)

        cv2.imshow('Label + LR + TD', cv2.hconcat([image, lr, td]))
        cv2.waitKey(0)
    cv2.destroyAllWindows()



def calculate_iou(gt_mask, pred_mask):
    """
    Manually calculate Intersection
    over Union

    @type   gt_mask : numpy.ndarray
    @param  gt_mask : the groundtruth mask

    @type   pred_mask : numpy.ndarray
    @param  pred_mask : the prediction mask
    """
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou


def get_color(mask, image):
    """
    Retrieve color of a pixel area.
    For each color in colors list, generate
    the relative mask (gt mask) and calculate intersection 
    over union between predicted mask and gt mask.

    Take the color that produces highest IOU

    @type   mask: numpy.ndarray
    @param  mask: The predicted mask used to retrieve the color of the pixel area
                  it covers

    @type   image: numpy.ndarray
    @param  image: The image on which predictions are made

    @rtype  col: str
    @return col: The color retrieved
    """
     # overlap_image == immagine con solo area pixel detectata dalla rete (maschera corrente), mantenendo colore immagine originale
    overlap_image = np.expand_dims(mask, axis=-1) * image.copy()
    mask[mask>0] = 255
    iou = 0
    col = ''
    for c in colors:
        gt_mask = generate_mask(overlap_image, c)
        gt_mask[gt_mask>0] = 255
        val = calculate_iou(mask, gt_mask)
        if val > iou:
            iou = val
            col = c
    return col



def get_info(image, prediction, draw = True):
    """
    Populate a dict with prediction informations,
    like label, bbox, center and mask data, to be
    later used for the metrics calculation

    Optionally, prediction data can be drawn on images for demo.

    @type   image: np.ndarray
    @param  image: the image to draw stuff

    @type   prediction: dict
    @param  prediction: predctions of the model

    @type   draw: bool
    @param  draw: whether draw data on images or not (for demo) 

    @rtype info: dict
    @return info: prediction data: label, bbox, mask, center, score
    """

    i = 0
    info = {"image": image, "predict": {}}
    for box in prediction[0]['boxes'].cpu()[:]:
        
        if prediction[0]['scores'].cpu().numpy()[i] < 0.85:
            continue
        box = np.uint(box.numpy())
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        pred_mask = (torch.squeeze(prediction[0]['masks'].cpu()).numpy()[i] > 0.5)
       
        col = get_color(pred_mask, image)
        
        if draw: 
            label = f"{col}_{list_class[str(prediction[0]['labels'].cpu().numpy()[i])]}" 
        else:
            label = f"{list_class[str(prediction[0]['labels'].cpu().numpy()[i])]}"
        x1, y1 = start_point
        x2, y2 = end_point
        center = (int((x1+x2)/2), int((y1+y2)/2))
        info['predict'][center] = (
            {"label": label,
             "start_point": start_point,
             "end_point": end_point,
             "center": center, 
             "mask": pred_mask, 
             "score": prediction[0]['scores'].cpu().numpy()[i]}
        )
        i += 1

    if draw:
        view(info)
    else:
        return info

        




