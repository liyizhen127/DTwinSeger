import os
import numpy as np
import cv2
from PIL import Image
import json
from metrics import IoUCalculator
import matplotlib.pyplot as plt

def mergeMasks(indexList, npyDir, idx):
    """Merge multiple masks into one"""
    mergedMask = None

    for index in indexList:
        npyFile = os.path.join(npyDir, f"segmentation_{index}.npy")
        if not os.path.exists(npyFile):
            raise FileNotFoundError(f"Mask file not found: {npyFile}")

        mask = np.load(npyFile)
        if mergedMask is None:
            mergedMask = np.zeros_like(mask, dtype=np.uint8)

        mergedMask = np.bitwise_or(mergedMask, mask.astype(np.uint8))
    

    return mergedMask


def calculateIntersectionRatio(bbox1, bbox2):
    """Calculate intersection over area ratio of two bounding boxes"""
    xMinInter = max(bbox1[0], bbox2[0])
    yMinInter = max(bbox1[1], bbox2[1])
    xMaxInter = min(bbox1[2], bbox2[2])
    yMaxInter = min(bbox1[3], bbox2[3])

    if xMinInter >= xMaxInter or yMinInter >= yMaxInter:
        return 0

    intersectionArea = (xMaxInter - xMinInter) * (yMaxInter - yMinInter)
    bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return intersectionArea / bbox1Area

def extract_objects_with_masks(image_path, outputJsonFile, segmentationDir):
        

    image = Image.open(image_path).convert("RGBA")
    image_np = np.array(image)


    with open(outputJsonFile, 'r') as f:
        masks_info = json.load(f)

    if not isinstance(masks_info, list):  
        masks_info = [masks_info]

    for mask_info in masks_info:
        index = mask_info["index"]
        segmentation_file = os.path.join(
            segmentationDir,
            f"mask_{index}.npy"
        )
        bbox = mask_info["bbox"] 


        mask = np.load(segmentation_file)
        if len(mask.shape) == 3:
            mask = mask[0] 

        mask = mask.astype(bool)

      
        xmin, ymin, xmax, ymax = map(int, bbox)
        cropped_image = image_np[ymin:ymax, xmin:xmax].copy()
        cropped_mask = mask[ymin:ymax, xmin:xmax]

     
        output_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)

    
        output_image[..., :3] = cropped_image[..., :3]  
        output_image[..., 3] = (cropped_mask * 255).astype(np.uint8)  

  
        final_image = Image.fromarray(output_image, mode='RGBA')


        output_path = os.path.splitext(segmentation_file)[0] + ".png"
        final_image.save(output_path)


def calculate_area(mask):
    return np.sum(mask)


def calculate_intersection(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2))


def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return

    import random
    random_anns = anns.copy()
    random.shuffle(random_anns)
    
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((random_anns[0]['segmentation'].shape[0], random_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in random_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
