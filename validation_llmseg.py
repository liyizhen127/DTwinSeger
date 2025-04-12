import torch
import gc
import os
import shutil  
import cv2  
from models.sam_model import SamProcessor
from models.owl_model import OwlSemanticDetector
from services.llm_service import LlmService
from utils.image_utils import loadImage
from utils.mask_utils import mergeMasks, extract_objects_with_masks
from metrics import IoUCalculator
import numpy as np
import os
from tqdm import tqdm
from dataloader.LLMSeg import LLMSeg
from metrics import IoUCalculator
from configs.config import ModelConfig, PathConfig
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
class Args:
    dataset_dir = r" "
    split = "val"  
    save_path = PathConfig.OUTPUT_DIR_LLMSEG_VAL
    json_path = ""
    coco_image_dir = ""

args = Args()
print(f"Loading LLMSeg-40K dataset for the {args.split} partition...")
reason_seg_data_dir = os.path.join(args.dataset_dir, args.split)

dataset = LLMSeg(
        json_path=args.json_path,
        coco_image_dir=args.coco_image_dir,
        num_samples=100,
        seed=70
    )
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
calculator = IoUCalculator(num_classes=2)
llmService = LlmService()
samProcessor = SamProcessor()
print("Starting inference...")
for idx, (query, image_path, mask_data) in enumerate(tqdm(dataset, desc="Processing Images")):
    imageOcv, imagePil = loadImage(image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    ext = os.path.splitext(image_name)[1]


    output_dir = args.save_path                                                
    output_image_path = os.path.join(  output_dir , image_name)
    shutil.copy(image_path, output_image_path)
    

    if isinstance(mask_data, np.ndarray):
        
  
        plt.figure(figsize=(10, 10))
        plt.imshow(imageOcv)
        

        mask_overlay = np.zeros_like(imageOcv)
        mask_overlay[mask_data > 0] = [255, 0, 0] 
        

        plt.imshow(mask_overlay, alpha=0.4)  
        

        plt.axis('off')
        

        gt_image_name = f"{base_name}_gt{ext}"
        gt_image_path = os.path.join(output_dir, gt_image_name)
        plt.savefig(gt_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        

        query_file_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(query_file_path, 'w', encoding='utf-8') as f:
            f.write(query)

        print(gt_image_path)
        print(query)
        

        seglevel=int(llmService.getLevel(query[0], image_path))

        outputJsonFile, outputMaskFile, final_mask = samProcessor.generateMasks(imageOcv, base_name, mask_data, 0, image_path, args.save_path) 

        llmService.getSemantic(outputJsonFile, image_path, outputMaskFile)
        final_mask = llmService.decideByLLM(outputJsonFile)
        calculator.update(final_mask, mask_data)
        ciou, giou = calculator.compute()
        print(f"cIoU: {ciou}, gIoU: {giou}")


