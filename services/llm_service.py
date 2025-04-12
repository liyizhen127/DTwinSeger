from openai import OpenAI
from configs.config import APIConfig
from configs.config import PromptConfig
from utils.image_utils import encodeImages
from utils.mask_utils import calculateIntersectionRatio
from utils.tensor_utils import convertTensorsToSerializable
from tqdm import tqdm
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import json


class LlmService:
    def __init__(self):
        self.client = OpenAI(
            base_url=APIConfig.OPENAI_BASE_URL,
            api_key=APIConfig.OPENAI_API_KEY
        )

    def getResponse(self, prompt, query, imageData, imageType="image/jpg"):
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

    def getConsistentResponse(self, prompt, query, imageData, num_queries=3, imageType="image/jpg"):
        responses = []
        
        for i in range(num_queries):
            response = self.getResponse(prompt, query, imageData, imageType)
            result = response.choices[0].message.content
            responses.append(result)
        
        consolidation_prompt = f"""You are given {num_queries} different responses to the same query.
Please analyze these responses and provide a consolidated answer that represents the best consensus.

Response 1:
{responses[0]}

Response 2:
{responses[1]}

Response 3:
{responses[2]}

Based on these responses, please provide the final consolidated answer."""
        
        final_prompt = prompt + query + "\n\n" + consolidation_prompt
        consensus_response = self.getResponse(final_prompt, "", imageData, imageType)
        
        return consensus_response.choices[0].message.content

    def getSemantic(self, json_path, image_path, output_mask_file, query):
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
            
            consensus_response = self.getConsistentResponse(
                PromptConfig.GET_SEMANTIC_PROMPT, 
                query, 
                imgB64StrList
            )
            
            new_item["semantic"] = consensus_response

            new_item.pop("segmentationFile", None)

            print(consensus_response)
            new_data.append(new_item)

        base_name = os.path.splitext(json_path)[0]
        new_json_path = base_name + "_final.json"

        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        return

