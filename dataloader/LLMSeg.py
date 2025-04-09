import torch
import json
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader

class LLMSeg(torch.utils.data.Dataset):

    def __init__(self, json_path, coco_image_dir=None, ego_objects_image_dir=None, num_samples=20, seed=42):
        self.json_path = json_path
        self.coco_image_dir = coco_image_dir
        self.ego_objects_image_dir = ego_objects_image_dir
        self.num_samples = num_samples
        self.seed = seed
        

        self.json_data = json.load(open(json_path, "r"))
        self.all_samples = self.load_all_samples()

        random.seed(self.seed)
        
        self.selected_indices = random.sample(range(len(self.all_samples)), min(num_samples, len(self.all_samples)))
        self.samples = [self.all_samples[i] for i in self.selected_indices]
    def __len__(self):
        return len(self.samples)
    
    def load_all_samples(self):
        samples = []
        data = self.json_data
        images = data.keys()
        
        for image in images:
            sample = data[image]
            from_dataset = sample['from_dataset']
            if from_dataset != 'coco':
                continue
        
            image_path = os.path.join(self.coco_image_dir, image)
                
            qas = sample['qa_pairs']
            for qa in qas:
                question = qa['question']

                rle_seg = qa['rle_seg']
                
                samples.append({
                    'image_path': image_path,
                    'question': question,
                    'from_dataset': from_dataset,
                    'rle_seg': rle_seg  
                })
        
        return samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['question']
        image_path = sample['image_path']
        

        try:
            rle_seg = sample['rle_seg']
      
            if not isinstance(rle_seg['counts'], bytes):
                rle_seg['counts'] = rle_seg['counts'].encode()
            
   
            mask = mask_utils.decode(rle_seg)
            

            mask = (mask > 0).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to decode mask: {e}")
 
            mask = np.zeros((480, 640), dtype=np.float32)
        
        return question, image_path, mask
