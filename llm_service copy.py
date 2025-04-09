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
        # 创建包含文本内容的消息
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt+query}]
        }]
        
        # 添加每张图片的数据
        for image in imageData:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{imageType};base64,{image}"
                }
            })
        
        # 调用 OpenAI API，传递多张图片
        return self.client.chat.completions.create(
            model=APIConfig.GPT_MODEL,
            messages=messages,
            max_tokens=2500
        )

    def getLevel(self, query, image_path):
        """获取图像的语义标签并将其添加到对应的 JSON 项中
        
        Args:
            json_path (str): JSON 文件路径。
            image_path (str): 输入图片路径。
            output_mask_file (str): 存储 segmentation_{index}.jpg 文件的文件夹路径。
            
        Returns:
            None
        """
        paths_list = [image_path]
        imgB64StrList = encodeImages(paths_list)
        # 获取响应，执行语义分析
        response = self.getResponse(PromptConfig.GET_LEVEL_PROMPT, query, imgB64StrList)
        elements = response.choices[0].message.content

        return elements
    def getSemantic(self, json_path, image_path, output_mask_file):
        """获取图像的语义标签并保存为新的 JSON 文件（去除 segmentationFile 字段）

        Args:
            json_path (str): 原始 JSON 文件路径。
            image_path (str): 输入图片路径。
            output_mask_file (str): segmentation_{index}.png 文件所在文件夹路径。

        Returns:
            None
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        new_data = []

        for item in tqdm(original_data, desc="正在处理 JSON 数据"):
            new_item = item.copy()

            paths_list = [image_path]
            idx = new_item.get("index")
            mask_path = os.path.join(output_mask_file, f"mask_{idx}.png")

            if os.path.exists(mask_path):
                paths_list.append(mask_path)
            else:
                print(f"警告: 找不到文件 {mask_path}")
                continue

            imgB64StrList = encodeImages(paths_list)
            response = self.getResponse(PromptConfig.GET_SEMANTIC_PROMPT, ' ', imgB64StrList)
            elements = response.choices[0].message.content

            new_item["semantic"] = elements

            # 删除 segmentationFile 字段
            new_item.pop("segmentationFile", None)

            print(elements)
            new_data.append(new_item)

        base_name = os.path.splitext(json_path)[0]
        new_json_path = base_name + "_final.json"

        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        return




    # def decideByLla(self, jsonPath, comment, imagePath):
    #     """Make decisions using LLA model

    #     Args:
    #         jsonPath: Path to JSON file with semantic information
    #         comment: User query/comment
    #         imagePath: Path to input image

    #     Returns:
    #         list: Selected element indices
    #     """
    #     imgB64Str = encodeImage(imagePath)

    #     with open(jsonPath, "r") as file:
    #         jsonData = json.load(file)
    #     jsonText = json.dumps(jsonData, indent=4)

    #     prompt = (f"{comment}\n\nYou need to provide the best matching element's index "
    #               f"based on the image and the question content I provide, combined with "
    #               f"the attributes of each element in the JSON file. "
    #               f"The output must be in the format: [number1, number2, ...] "
    #               f"only reply me the format of [number1, number2, ...] the min of item is 1 . Here is the JSON content:\n{jsonText}")

    #     response = self.getResponse(prompt, imgB64Str)
    #     print(response)
    #     return response.choices[0].message.content.strip("[]").split(", ")