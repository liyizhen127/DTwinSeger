import numpy as np
import os
import json
import random
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from configs.config import ModelConfig, PathConfig
from utils.mask_utils import calculate_area, calculate_intersection, show_mask, show_anns
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import label
from utils.mask_utils import extract_objects_with_masks

class SamProcessor:
    """SAM model processor"""

    def __init__(self):
        sys.path.append("..")
        sam_checkpoint = "" 
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device='cuda:7')

        self.maskGenerator = SamAutomaticMaskGenerator(sam, points_per_batch=2)
        self.maskPredictor = SamPredictor(sam)

    def processSegmentation(self, mask, idx, segmentationDir):

        bbox = mask.get("bbox")
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        xMin, yMin, width, height = bbox
        xMax, yMax = xMin + width, yMin + height

        predicted_iou = mask.get('predicted_iou')

        segmentation = mask.get("segmentation")
        if segmentation is None:
            return None

        segmentationFile = os.path.join(
            segmentationDir,
            f"mask_{idx}.npy"
        )
        np.save(segmentationFile, segmentation)
        
        return {
            "index": idx,
            "bbox": [xMin, yMin, xMax, yMax],
            "predicted_iou":predicted_iou,
            "semantic": " "
        }

        
    def filter_and_merge_masks(self, masks, mask_data, threshold=0.7):
        if len(masks) == 0:
            return np.zeros_like(mask_data, dtype=bool), []
        

        final_mask = np.zeros_like(masks[0]['segmentation'], dtype=bool)
        selected_masks = []
        

        mask_data_bool = mask_data.astype(bool)
        

        for i, mask_info in enumerate(masks):
            mask = mask_info['segmentation']
            

            intersection = np.logical_and(mask, mask_data_bool)
            intersection_area = np.sum(intersection)
            mask_area = np.sum(mask)
            
  
            if mask_area > 0 and intersection_area / mask_area > threshold:
                final_mask = np.logical_or(final_mask, mask)
                selected_masks.append(i)
                
        return final_mask, selected_masks
    def generate_bbox(self, mask):

   
        ys, xs = np.where(mask > 0)

        if ys.size == 0 or xs.size == 0:
            return None
        

        ymin = np.min(ys)
        ymax = np.max(ys)
        xmin = np.min(xs)
        xmax = np.max(xs)
        
        return (xmin, ymin, xmax, ymax)

    def generateMasks(self, image, image_name, mask_data, seglevel, image_path, save_path):

        

        outputJsonFile = os.path.join(save_path, f"{image_name}.json")
        if os.path.exists(outputJsonFile):
            return outputJsonFile, os.path.join(save_path, f"{image_name}"), [1]
        

        masks = self.maskGenerator.generate(image)
        

        segmentationDir = os.path.join(save_path, f"{image_name}")
        os.makedirs(segmentationDir, exist_ok=True)

        origin_path = os.path.join(segmentationDir, f"{image_name}.jpg")
        cv2.imwrite(origin_path, image)

        final_mask, selected_masks = self.filter_and_merge_masks(masks, mask_data)
        
        filtered_masks = [{
                'segmentation': final_mask,
                'bbox': self.generate_bbox(final_mask)
            }]
        filtered_masks.extend([masks[i] for i in range(len(masks)) if i not in selected_masks])

        i = 1
        while i < len(filtered_masks):
            next_mask = filtered_masks[i]['segmentation']
            intersection = np.logical_and(filtered_masks[0]['segmentation'], next_mask).sum()
            next_area = np.sum(next_mask)
            overlap = intersection / next_area
            if overlap > 0.8:
                del filtered_masks[i]
            i+=1

        if seglevel == 0:
            i = 1
            while i < len(filtered_masks):
                current_mask = filtered_masks[i]['segmentation']

                coords = np.argwhere(current_mask)
                if coords.size == 0:
                    i += 1
                    continue

                center = np.mean(coords, axis=0).astype(int)
                input_point = np.array([[center[1], center[0]]])

                input_label = [1]
                self.maskPredictor.set_image(image)

                masks_multi, _, _ = self.maskPredictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )

                areas = [np.sum(m) for m in masks_multi]
                if not areas:
                    i += 1
                    continue

                max_idx = np.argmax(areas)
                max_mask = masks_multi[max_idx]
                filtered_masks[i] = {
                'segmentation': max_mask,
                'bbox': self.generate_bbox(max_mask)
            }

                j = i + 1
                while j < len(filtered_masks):
                    next_mask = filtered_masks[j]['segmentation']
                    intersection = np.logical_and(max_mask, next_mask).sum()
                    next_area = np.sum(next_mask)
                    if next_area == 0:
                        j += 1
                        continue

                    overlap = intersection / next_area
                    if overlap > 0.8:
                        del filtered_masks[j]
                    else:
                        j += 1  

                i += 1

            i = 1
            while i < len(filtered_masks):
                next_mask = filtered_masks[i]['segmentation']
                intersection = np.logical_and(filtered_masks[0]['segmentation'], next_mask).sum()
                next_area = np.sum(next_mask)
                overlap = intersection / next_area
                if overlap > 0.75:
                    del filtered_masks[i]
                    continue
                i+=1
            i = 1
            while i < len(filtered_masks):
                mask_i = filtered_masks[i]['segmentation']
                area_i = np.sum(mask_i)

                j = i + 1
                while j < len(filtered_masks):
                    mask_j = filtered_masks[j]['segmentation']
                    area_j = np.sum(mask_j)

                    intersection = np.logical_and(mask_i, mask_j).sum()
                    if intersection == 0:
                        j += 1
                        continue

                    overlap_i = intersection / area_i
                    overlap_j = intersection / area_j
                    if max(overlap_i, overlap_j) > 0.85:
                        if area_i > area_j:
                            del filtered_masks[i]
                            break  
                        else:
                            del filtered_masks[j]
                            continue  

                    j += 1
                else:
                    i += 1

        else:

            base_mask = filtered_masks[0]['segmentation']
            base_area = calculate_area(base_mask)

            i = 1
            while i < len(filtered_masks):
                inter_area = calculate_intersection(base_mask, filtered_masks[i]['segmentation'])
                if inter_area / base_area > 0.8:
                    del filtered_masks[i]
                else:
                    i += 1

        processedData = []
        for i, mask in enumerate(filtered_masks):
            result = self.processSegmentation(mask, i, segmentationDir)
            if result:
                processedData.append(result)
        with open(outputJsonFile, "w") as jsonFile:
            json.dump(processedData, jsonFile, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else o)
          
        return outputJsonFile, segmentationDir, final_mask
