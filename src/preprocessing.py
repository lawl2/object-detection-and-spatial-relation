import numpy as np
import cv2
import json
import os
import glob
import re


list_class = {'blue_sphere': '0',
              'blue_cube': '1',
              'blue_cylinder': '2',
              'yellow_sphere': '3',
              'yellow_cube': '4',
              'yellow_cylinder': '5',
              'brown_sphere': '6',
              'brown_cube': '7',
              'brown_cylinder': '8',
              'red_sphere': '9',
              'red_cube': '10',
              'red_cylinder': '11',
              'cyan_sphere': '12',
              'cyan_cube': '13',
              'cyan_cylinder': '14',
              'green_sphere': '15',
              'green_cube': '16',
              'green_cylinder': '17',
              'gray_sphere': '18',
              'gray_cube': '19',
              'gray_cylinder': '20',
              'purple_sphere': '21',
              'purple_cube': '22',
              'purple_cylinder': '23'}


def parse_args(return_parser=False):
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing on data"
    )
    parser.add_argument("--train", default=False, action="store_true")
    args = parser.parse_args()
    return args




def remove_images(train, bbJsonPath):
    """
    Remove  bad images from test/train data.
    Bad images are the ones that has
    two object with same color and material;
    in this cases, the same mask is generated,
    and it covers the area of both objects.

    This function check if there is the same mask
    for more than one object, and it moves
    images and relatively generated masks 
    in another folder.

    @type   train: boolean  
    @param  train: True if this step needs to be done on train test

    @type   bbJsonPath: str
    @param  bbJsonPath: The path to the JSON file containing bbox informations
 
    """

    if train:
        fold = 'train'
    else:
        fold = 'test'
    img_dir = 'data/' + fold + '/images/'
    mask_dir = 'data/'+ fold +'/masks/'
    rem_img = 'data/'+ fold + '/rem_imgs/'
    rem_mask = 'data/' + fold + '/rem_masks/'

    
    imgs = list(sorted(os.listdir(img_dir)))
    dup_mask = []
    with open(bbJsonPath, 'r') as jf:
        json_file = json.load(jf)

    # iterate over each image
    for img in imgs:
        # take info only for the current image
        mask_info = [j for j in json_file if os.path.splitext(img)[0] in j['filename']]

        for f in mask_info:
            i = 0
            split = f['filename'].split("_")
            if (f"{split[0]}{split[1]}{split[2]}.png") in dup_mask:
                continue

            for j in mask_info:
                if f['bbox'] == j['bbox'] or f['bbox'] == [[0, 0], [0, 0]] or j['bbox'] == [[0, 0], [0, 0]] :
                    i += 1
                    if i >= 2:
                        splitted = f['filename'].split("_")
                        if f"{splitted[0]}_{splitted[1]}_{splitted[2]}.png" not in dup_mask:
                            dup_mask.append(f"{splitted[0]}_{splitted[1]}_{splitted[2]}.png")
                            print(f)
                            break
    print(dup_mask)
    masks = list(sorted(os.listdir(mask_dir)))
    for rem in dup_mask:
        os.rename(img_dir + rem, rem_img + rem)
        for m in masks:
            if os.path.splitext(rem)[0] in m:
                print(rem_mask + m)
                os.rename(mask_dir + m, rem_mask + m) 


def generate_mask(image, color):
    """
    Generate a labeled mask.
    First generate a pixel mask of the 
    specified color out of the image,
    then use connected components
    algorithm to label the image.

    A bilateral filter is applied to the image
    to smooth reflections on objects in the scenes
    while mantaining sharp edges

    @type   image: numpy.ndarray 
    @param  image: the image used for extracting the mask

    @type   color: str
    @param  color: the color to select the mask

    @rtype : numpy.ndarray
    @returns labels : a matrix with labels  
    """
    global color_mask
    global hsv
    img = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Different values of bilateral filter if the object is gray
    if color == 'gray':
        seed = (10,10)
        img = cv2.bilateralFilter(img, 11, 37, 99)
        cv2.floodFill(img, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(3, 3, 3, 3), upDiff=(3, 3, 3, 3))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 0])
        upper_gray = np.array([360, 40, 255])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=1)

    else:
        img = cv2.bilateralFilter(img, 1, 75, 75)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 85, 0])
        upper_gray = np.array([360, 255, 255])
        lower_gray2 = np.array([0, 0, 255])
        upper_gray2 = np.array([360, 255, 255])
        mask1 = cv2.inRange(hsv, lower_gray, upper_gray)
        mask2 = cv2.inRange(hsv, lower_gray2, upper_gray2)
        mask = mask1 + mask2
        img = cv2.bitwise_and(img, img, mask=mask)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == 'red':
        # lower red
        lower_red = np.array([0, 50, 0])
        upper_red = np.array([10, 255, 255])
        # upper red
        lower_red2 = np.array([170, 50, 0])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res1 = cv2.bitwise_and(img, img, mask=mask)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        res2 = cv2.bitwise_and(img, img, mask=mask2)
        res = res1 + res2
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'blue':
        lower_blue = np.array([100, 50, 0])
        upper_blue = np.array([120, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'brown':
        lower_brown = np.array([11, 50, 0])
        upper_brown = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'yellow':
        lower_yellow = np.array([20, 50, 0])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'green':
        lower_green = np.array([40, 50, 0])
        upper_green = np.array([85, 256, 256])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'purple':
        lower_purple = np.array([120, 50, 0])
        upper_purple = np.array([150, 256, 256])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    if color == 'cyan':
        lower_cyan = np.array([80, 50, 0])
        upper_cyan = np.array([100, 256, 256])
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel, iterations=1)

    bil_img = cv2.bilateralFilter(res, 5, 105, 105)
    color_mask = cv2.cvtColor(bil_img, cv2.COLOR_RGB2GRAY)

    # Eliminate Small Edges
    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
    hist = np.zeros(num_labels)
    lab = labels.copy().flatten()

    for j in range(0, num_labels):
        hist[j] = np.count_nonzero(lab == j)
        # if hist[j] < 162:
        if hist[j] < 1:
            labels[labels == j] = 0
    return labels


def save_masks(imgsPath, jsonPath, maskImgsPath):
    """
    Save binary masks in a folder.
    Iterates over all the colors and all the objecs:
    for each object generate 
    a pixel mask, then save the mask in a folder.

    A JSON file containing meta data is used to 
    iterates over all the objects in the image.

    @type   imgsPath: str
    @param  imgsPath: The images folder location

    @type   jsonPath: str
    @param  jsonPath: The json file folder location
    
    @type   maskImgsPath: str
    @param  maskImgsPath: The path to save generated binary masks
    """
    # read json file
    with open(jsonPath, 'r') as f:
        data = json.load(f)

    # iterate on every image
    for ip in imgsPath:
        filename = os.path.basename(ip)
        f_name, ext = os.path.splitext(filename)
        img_info = [val for val in data.get('scenes') if val.get("image_filename") == filename]
        print(ip)
        if len(img_info):
            img_info = img_info[0]

        # read Image
        img = cv2.imread(ip, 1)
        colors = ['gray', 'red', 'blue', 'brown', 'yellow', 'green', 'purple', 'cyan']
        for color in colors:
            mask = generate_mask(img, color)
            
            # get information from json based on color
            obj_color = [val for val in img_info.get('objects') if val.get("color") == color]
            for val in obj_color:
                temp = np.zeros(img.shape[0:2])
                y, x = val.get('pixel_coords')[0:2]
                shape = val.get('shape')
                temp_filename = f"{f_name}_{list_class[f'{color}_{shape}']}_{x}_{y}{ext}"

                label = mask[x, y]
                
                temp += mask == label
                if temp.sum() > (img.shape[0] * img.shape[1] / 2):
                    temp = -1 * temp + temp.max()

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=5)
                temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel, iterations=2)
                cv2.imwrite(f"{maskImgsPath}/{temp_filename}", temp, [cv2.IMWRITE_PNG_BILEVEL, 1])



def select_nearest_mask(maskImgsPath):
    """
    Process only images that has the same mask
    for more than one object.
    This can happen when two or more objects of same 
    color and material are close each other.

    Firstly it generates contours for each object, 
    then iterate.
    For each saved mask calculates the distance between
    its center and the contour, and take the minimum distance.
    Finally save an image with only the mask with minimum distance from the contour.

    @type   maskImgsPath: str
    @param  maskImgsPath: The location of saved binary masks
    """
    i = 0
    for img_path in glob.glob(f"{maskImgsPath}*"):
        
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        num_labels , labels = cv2.connectedComponents(image, connectivity=8)
        # if only 1 object and background --> no need to process
        if num_labels < 3:
            continue

        # one image for each object 
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x, y = [int(i) for i in re.findall(r'[0-9]+', img_path)[2:]]

        min = None
        min_contour = None
        # iterate on every object contour
        for cnt in contours:
            dist = abs(cv2.pointPolygonTest(cnt, (y, x), True))
            if min is None or dist < min:
                min = dist
                min_contour = cnt
        
        temp = np.zeros(image.shape[:2])
        try:
            temp += labels == labels[min_contour[1, 0, 1], min_contour[1, 0, 0]]
        except:
            print(img_path)
        cv2.imwrite(f"{maskImgsPath}{filename}", temp, [cv2.IMWRITE_PNG_BILEVEL, 1])


def generate_bbox(maskImgsPath, bbJsonPath):
    """
    Generate bounding boxes from 
    prevoiusly generated masks.
    Saves coordinates of the four point
    of the bounding box in a JSON file.

    @type   bbJsonPath: str
    @param  bbJsonPath: The location of bb info file

    @type   maskImgsPath: str
    @param  maskImgsPath: The location of saved binary masks
    """
    i = 0
    temp = []
    
    for file in glob.glob(f"{maskImgsPath}*"):
        i += 1
        if i % 100 == 0:
            print(f"{i} {file}")
        ima = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        x, y, w, h = cv2.boundingRect(ima)

        filename = os.path.basename(file)
        box = (x, y), (x + w, y + h)
        temp.append({"filename": filename, "bbox": box})

    with open(bbJsonPath, 'w') as fp:
        json.dump(temp, fp)


def preprocessing(args):
    if args.train:
        imgsPath = [f for f in glob.glob("data/train/images/*.png")]
        jsonPath = 'data/CLEVR_train_scenes.json'
        maskPath = 'data/train/masks/'
        bbJsonPath = 'data/CLEVR_train_bbox.json'
    
    else:
        imgsPath = [f for f in glob.glob("data/test/images/*.png")]
        jsonPath = 'data/CLEVR_test_scenes.json'
        maskPath = 'data/test/masks/'
        bbJsonPath = 'data/CLEVR_test_bbox.json'        
    
    save_masks(imgsPath, jsonPath, maskPath)
    select_nearest_mask(maskPath)
    generate_bbox(maskPath, bbJsonPath)
    remove_images(args.train ,bbJsonPath)


if __name__ == "__main__":
    preprocessing(parse_args())



