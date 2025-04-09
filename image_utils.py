import base64
import cv2
from PIL import Image
import os
import numpy as np
import json

def encodeImage(imagePath):
    """Convert a single image to Base64 encoding"""
    with open(imagePath, "rb") as imageFile:
        return base64.b64encode(imageFile.read()).decode('utf-8')

def encodeImages(imagePaths):
    """Convert multiple images to Base64 encoding"""
    return [encodeImage(imagePath) for imagePath in imagePaths]

def loadImage(imagePath):
    """Load image in both OpenCV and PIL format"""
    imageOcv = cv2.imread(imagePath)
    imageOcv = cv2.cvtColor(imageOcv, cv2.COLOR_BGR2RGB)
    imagePil = Image.open(imagePath)
    return imageOcv, imagePil

def crop_images_from_json(outputJsonFile, imagePath, outputMaskFile):

    with open(outputJsonFile, 'r') as f:
        data = json.load(f)
    image = np.array(Image.open(imagePath))
    

    os.makedirs(outputMaskFile, exist_ok=True)
    
    for item in data:
        idx = item['index']

        xmin, ymin, xmax, ymax = map(int, item['bbox'])
        

        mask_path = os.path.join(outputMaskFile, f"segmentation_{idx}.npy")
        if not os.path.exists(mask_path):
            continue
            
   
        cropped_img = image[ymin:ymax, xmin:xmax]

        mask = np.load(mask_path)[ymin:ymax, xmin:xmax]
        
   
        cropped_img[mask == 0] = 0
        
   
        output_path = os.path.join(outputMaskFile, f"segmentation_{idx}.jpg")
        if os.path.exists(output_path):
            continue
        
  
        Image.fromarray(cropped_img).save(output_path)

def create_image_paths(json_path, image_path, output_mask_file):

    with open(json_path, 'r') as f:
        data = json.load(f)


    paths_list = [image_path]

 
    for item in data:

        idx = item["index"]

   
        segmentation_image_path = os.path.join(output_mask_file, f"segmentation_{idx}.jpg")


        if os.path.exists(segmentation_image_path):
            paths_list.append(segmentation_image_path)
        else:
            print(f"Warning: No {segmentation_image_path}")

    return paths_list

def draw_bboxes_on_image(image_path, bboxes_dict, output_path):


    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    

    for label, bbox_data in bboxes_dict.items():
 
        box = bbox_data["box"]
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        
        color = (0, 255, 0)  
        thickness = 2
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        

        score = bbox_data["score"]
        label_text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)  
        text_thickness = 1
       
        text_size = cv2.getTextSize(label_text, font, font_scale, text_thickness)[0]
        text_x = x_min
        text_y = y_min - 5  
        
  
        cv2.rectangle(image, 
                     (text_x, text_y - text_size[1]),
                     (text_x + text_size[0], text_y + 5),
                     (0, 0, 0),
                     -1) 
        
     
        cv2.putText(image, label_text, (text_x, text_y),
                    font, font_scale, text_color, text_thickness)
  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, image)
    
    return output_path
    
