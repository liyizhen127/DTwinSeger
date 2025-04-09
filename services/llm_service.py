from openai import OpenAI
from configs.config import APIConfig
from configs.config import PromptConfig
from utils.image_utils import encodeImages
from utils.mask_utils import calculateIntersectionRatio
from utils.tensor_utils import convertTensorsToSerializable
from tqdm import tqdm
import json
import os

class LlmService:
    """Service for LLM API interactions and semantic processing"""

    def __init__(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(
            base_url=APIConfig.OPENAI_BASE_URL,
            api_key=APIConfig.OPENAI_API_KEY
        )

    def getResponse(self, prompt, query, imageData, imageType="image/jpg"):
        """Get response from LLM model with multiple image inputs

        Args:
            prompt: Text prompt for the model
            imageData: List of Base64 encoded image data
            imageType: Type of image data

        Returns:
            OpenAI API response
        """
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt+query}]
        }]
        
        for image in imageData:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{imageType};base64,{image}"
                }
            })
        
        return self.client.chat.completions.create(
            model=APIConfig.GPT_MODEL,
            messages=messages,
            max_tokens=2500
        )

    def getLevel(self, query, image_path):
        paths_list = [image_path]
        imgB64StrList = encodeImages(paths_list)

        response = self.getResponse(PromptConfig.GET_LEVEL_PROMPT, query, imgB64StrList)
        elements = response.choices[0].message.content

        return elements
    def getSemantic(self, json_path, image_path, output_mask_file):
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        new_data = []

        for item in tqdm(original_data, desc="loading"):
            new_item = item.copy()

            paths_list = [image_path]
            idx = new_item.get("index")
            mask_path = os.path.join(output_mask_file, f"mask_{idx}.png")

            if os.path.exists(mask_path):
                paths_list.append(mask_path)
            else:
                print(f"Warning!No {mask_path}")
                continue

            imgB64StrList = encodeImages(paths_list)
            response = self.getResponse(PromptConfig.GET_SEMANTIC_PROMPT, ' ', imgB64StrList)
            elements = response.choices[0].message.content

            new_item["semantic"] = elements

            new_item.pop("segmentationFile", None)

            print(elements)
            new_data.append(new_item)

        base_name = os.path.splitext(json_path)[0]
        new_json_path = base_name + "_final.json"

        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        return

