import cv2
import numpy as np
import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import ClevrDataset
from utils import get_transform, get_model, get_info


list_class = { '0' : 'blue_sphere',
               '1' : 'blue_cube',
               '2' : 'blue_cylinder' ,
              '3' : 'yellow_sphere',
              '4' : 'yellow_cube',
              '5' : 'yellow_cylinder',
              '6' : 'brown_sphere',
              '7' : 'brown_cube',
              '8' : 'brown_cylinder',
              '9' : 'red_sphere',
              '10' : 'red_cube',
              '11' : 'red_cylinder',
              '12' : 'cyan_sphere',
              '13' : 'cyan_cube',
              '14' : 'cyan_cylinder',
              '15' : 'green_sphere',
              '16' : 'green_cube',
              '17' : 'green_cylinder',
              '18' : 'gray_sphere',
              '19' : 'gray_cube',
              '20' : 'gray_cylinder',
              '21' : 'purple_sphere',
              '22' : 'purple_cube',
              '23' : 'purple_cylinder'}

classes = { 'sphere': 1,
            'cube'  : 2,
            'cylinder' : 3}

colors = ['gray', 'red', 'blue', 'brown', 'yellow', 'green', 'purple', 'cyan']


def parse_args(return_parser=False):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pytorch Object Detection and Spatial Reasoning"
    )
    parser.add_argument("--demo", default=False, action="store_true")
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--class_metrics", default=False, action="store_true")
    args = parser.parse_args()
    return args


def calc_metrics(info, target, metric):
    """
    Calculate MAP (Mean Average Precision) for one image.
    Populates dictionaries gt, pred in order to compare
    predicted labels vs gt labels.

    @type   info: dict
    @param  info: dictionary with prediction data (scores, bboxes, masks) 

    @type   target: dict
    @param  target: dictionary with ground-truth data

    @type   metric: torchmetrics.detection.mean_ap.MeanAveragePrecision
    @param  metric: metric calculator

    @rtype  results: dict
    @return results: dictionary with calculated metrics

    """
    # dict used to calculate metrics
    gt = {}
    pred = {}

    # retrieve gt and pred centers and sort by x axis(from left to right)
    gt_centers = []
    pred_centers = []
    for i in target['centers']:
        gt_centers.append(i)
    for i in info['predict'].keys():
        pred_centers.append(i)
    gt_centers.sort()
    pred_centers.sort()

    # populate gt dict
    for i in gt_centers:
        for j in range(len(target['centers'])):
            if i == target['centers'][j]:
                shape = list_class[str(int(target['labels'][j].cpu().numpy()))].split("_")[1]
                c = classes[shape]
                gt[i] = {
                    'bbox': target['boxes'][j],
                    'label': torch.Tensor(np.array([c]))
                }

    # populate pred dict
    bbox = np.zeros(4)
    label = np.zeros(1)
    score = np.zeros(1)
    for i in pred_centers:
        for k in info['predict'].keys():
            if i == k:
                bbox = (
                    info['predict'][k]['start_point'][0],
                    info['predict'][k]['start_point'][1],
                    info['predict'][k]['end_point'][0],
                    info['predict'][k]['end_point'][1]
                )
                label = int(classes[info['predict'][k]['label']])
                score = info['predict'][k]['score']
                pred[i] = {
                    'bbox' : torch.as_tensor(bbox, dtype=float),
                    'label': torch.as_tensor(label, dtype=int),
                    'score': torch.as_tensor(score, dtype=float),
                }
    
    # get full size predicion tensors to calculate metrics
    p_bbox = torch.FloatTensor(size=(len(pred.keys()), 4))
    p_labels = torch.IntTensor(size=(len(pred.keys()),))
    p_scores = torch.FloatTensor(size=(len(pred.keys()),))
    j = 0
    for i in pred.keys():
        p_bbox[j] = pred[i]['bbox']
        p_labels[j] = pred[i]['label']
        p_scores[j] = pred[i]['score']
        j += 1
    p = [
        dict (
            boxes=p_bbox,
            scores=p_scores,
            labels=p_labels,
        )
    ]

    # get full size ground truth tensors to clauclate metrics
    g_bbox = torch.FloatTensor(size=(len(gt.keys()), 4))
    g_labels = torch.IntTensor(size=(len(gt.keys()),))
    j = 0
    
    for i in gt.keys():
        g_bbox[j] = gt[i]['bbox']
        g_labels[j] = gt[i]['label']
        j += 1
    t = [
        dict (
            boxes=g_bbox,
            labels=g_labels
        )
    ]
    
    # update metrics
    metric.update(preds=p, target=t)
    results = metric.compute()
    for k in results.keys():
        results[k] = results[k]

    return results


def test(dataset, model, device, demo, class_metrics=False):
    """
    Open a text file and write into it test metrics,
    or run a demo which display predictions on images.

    @type   dataset: torch.utils.data.Dataset
    @param  dataset: The dataset on which do predictions

    @type   model: torchvision.models.detection.mask_rcnn.MaskRCNN
    @param  model: MaskRCNN used to do predictions

    @type   device: torch.device
    @param  device: CPU or GPU, depends on the machine

    @type   demo: boolean
    @param  demo: True if run in demo mode(no results, just display)

    @type   class_metrics: boolean
    @param  class_metrics: True for enabling calculating metrics per class
    """
    if class_metrics:
        f = open("test_results_per_class.txt", "w+")
    else:
        f = open("test_reults.txt", "w+")

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=class_metrics)

    for i in range(len(dataset.imgs)):
        img, target = dataset[i]
        model.to(device)
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
            image = cv2.cvtColor(
            np.uint8(img.mul(255).permute(1, 2, 0).byte().numpy().copy()),
            cv2.COLOR_BGR2RGB
        )

        if demo:
            get_info(image, prediction, True)
        else:
            info = get_info(image, prediction, False)
            results = calc_metrics(info, target, metric)
            if i%10==0:
                print(f"iterazione: {i}")
                f.write(str(results))
                f.write("\n")
    f.close()


def main(args):
    num_classes = 4
    model = get_model(num_classes)
    model.load_state_dict(torch.load("data/model/weights_customj4_1_BCK_2022_11_07_00.torch"))
    
    print("\n\nMODEL LOADED\n\n")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = ClevrDataset(get_transform(train=False), test=True ,test_size=args.n)
    test(dataset, model, device, args.demo, args.class_metrics)



if __name__ == "__main__":
    main(parse_args())