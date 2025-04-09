import torch
import gc
import os
from models.sam_model import SamProcessor
from models.owl_model import OwlSemanticDetector
from services.llm_service import LlmService
from utils.image_utils import loadImage
from utils.mask_utils import mergeMasks, draw_masks_on_image, extract_objects_with_masks
from metrics import IoUCalculator
import numpy as np
import os
from tqdm import tqdm
from dataloader.ReferSeg import ReferSegDataset
from metrics import IoUCalculator
import matplotlib.pyplot as plt
import shutil  
class Args:
    refcoco_path = r" "  
    split = "val"  

args = Args()
print(f"Loading ReferSeg dataset for the {args.split} partition...")

dataset = ReferSegDataset(args.refcoco_path, args.split, samples_per_epoch=100) 
calculator = IoUCalculator(num_classes=2)
llmService = LlmService()
samProcessor = SamProcessor()
calculator = IoUCalculator(num_classes=2)
print("Starting inference...")
for idx, (image_path, mask_data, query) in enumerate(tqdm(dataset, desc="Processing Images")):

        query = query[0]
        imageOcv, imagePil = loadImage(image_path)
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1]

  
        output_dir = args.save_path
        output_image_path = os.path.join(  output_dir , image_name)
        shutil.copy(image_path, output_image_path)
        
 
        if isinstance(mask_data, np.ndarray):
            
            
            plt.figure(figsize=(10, 10))
            composite = imageOcv.copy()

     
            mask = mask_data > 0
            composite[mask] = imageOcv[mask] * 0.6 + np.array([255, 0, 0]) * 0.4

            plt.imshow(composite)
            plt.axis('off')
            

            gt_image_name = f"{base_name}_our{ext}"
            gt_image_path = os.path.join(output_dir, gt_image_name)
            plt.savefig(gt_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        
  
        query_file_path = os.path.join(output_dir, f"{base_name}.txt")
        np.save(os.path.join(output_dir, f"{base_name}_gt.npy"),mask_data)
        with open(query_file_path, 'w', encoding='utf-8') as f:
            f.write(query)

        print(gt_image_path)
        print(query)
        

        seglevel=int(llmService.getLevel(query[0], image_path))

        outputJsonFile, outputMaskFile, final_mask = samProcessor.generateMasks(imageOcv, base_name, mask_data, 0, image_path, args.save_path) 
        np.save(os.path.join(output_dir, f"{base_name}_our.npy"),final_mask)

        plt.figure(figsize=(10, 10))
        composite = imageOcv.copy()


        mask = mask_data > 0
        composite[mask] = imageOcv[mask] * 0.6 + np.array([255, 0, 0]) * 0.4

        plt.imshow(composite)
        plt.axis('off')

        gt_image_name = f"{base_name}_gt{ext}"
        gt_image_path = os.path.join(output_dir, gt_image_name)
        plt.savefig(gt_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        llmService.getSemantic(outputJsonFile, image_path, outputMaskFile)
        indices = llmService.decideByLLM(outputJsonFile)
        calculator.update(indices, mask_data)
        ciou, giou = calculator.compute()
        print(f"cIoU: {ciou}, gIoU: {giou}")