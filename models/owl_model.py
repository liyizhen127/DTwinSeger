import torch

# from transformers import Owlv2Processor, Owlv2ForObjectDetection
from configs.config import ModelConfig
import json
import os
from utils.tensor_utils import convertTensorsToSerializable
from utils.mask_utils import calculateIntersectionRatio
from utils.image_utils import draw_bboxes_on_image

class OwlSemanticDetector:

    def __init__(self):
        self.tokenizer = Owlv2Processor.from_pretrained(ModelConfig.OWL_MODEL_PATH)
        self.model = Owlv2ForObjectDetection.from_pretrained(ModelConfig.OWL_MODEL_PATH)
        if torch.cuda.is_available():
            self.model.to(ModelConfig.DEVICE)

    def predictHighestScoreBoxes(self, texts, image, idx):

        # Prepare inputs
        inputs = self.tokenizer(text=texts, images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(ModelConfig.DEVICE)
        
        # Get model predictions
        outputs = self.model(**inputs)
        targetSizes = torch.Tensor([image.size[::-1]])
        results = self.tokenizer.post_process_object_detection(
            outputs=outputs,
            target_sizes=targetSizes,
            threshold=0.1
        )
        
        # Extract highest scoring boxes
        text = texts[0]
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"]
        )
        
        base_path = " "
        temp_dir = os.path.join(base_path, "temp")
        temp_image_path = os.path.join(temp_dir, f"original_{idx}.jpg")
        os.makedirs(temp_dir, exist_ok=True)
        
        image.save(temp_image_path)
        
        highestScoreBoxes = {}
        for box, score, label in zip(boxes, scores, labels):
            label_text = text[label]
            if (label_text not in highestScoreBoxes or
                    score > highestScoreBoxes[label_text]["score"]):
                highestScoreBoxes[label_text] = {
                    "box": box,
                    "score": score,
                    "confidence": float(score)
                }
        
        if highestScoreBoxes:
            vis_output_path = os.path.join(base_path, f"{idx}_all_bboxes.jpg")
            try:
                draw_bboxes_on_image(temp_image_path, highestScoreBoxes, vis_output_path)
            except Exception as e:
                print(f"Visualization failed: {str(e)}")
        
        try:
            os.remove(temp_image_path)
        except Exception as e:
            print(f"Failed to remove temporary image: {str(e)}")
        return highestScoreBoxes

    def updateSemanticInfo(self, jsonFilePath, highestScoreBoxes, threshold=0.2):
        with open(jsonFilePath, "r") as jsonFile:
            data = json.load(jsonFile)
            
   
        highestBoxesWithLabels = [
            {
                "box": boxData["box"],
                "label": label,
                "confidence": boxData["confidence"]
            }
            for label, boxData in highestScoreBoxes.items()
        ]

    
        updatedData = []
        for element in data:
            elementBox = element["bbox"]
            for highestBox in highestBoxesWithLabels:
                if calculateIntersectionRatio(elementBox, highestBox["box"]) > threshold:
                    element["semantic"] = highestBox["label"]
                    element["semantic_confidence"] = highestBox["confidence"]
                    updatedData.append(element)
                    break
        print('----')
        print(len(updatedData))
   
        updatedData = convertTensorsToSerializable(updatedData)
        with open(jsonFilePath, "w") as jsonFile:
            json.dump(updatedData, jsonFile, indent=4)
