import os
import cv2
import json
import numpy as np
import torch


class ReasonSegDataset(torch.utils.data.Dataset):

    def __init__(self, base_image_dir):

        self.base_image_dir = base_image_dir


        self.images = [
            os.path.join(base_image_dir, f)
            for f in os.listdir(base_image_dir)
            if f.endswith(".jpg")
        ]

    def __len__(self):

        return len(self.images)
 

    def get_mask_from_json(self,json_path, img):
        try:
            with open(json_path, "r") as r:
                anno = json.loads(r.read())
        except:
            with open(json_path, "r", encoding="cp1252") as r:
                anno = json.loads(r.read())

        inform = anno["shapes"]
        comments = anno["text"]
        is_sentence = anno["is_sentence"]

        height, width = img.shape[:2]

  
        area_list = []
        valid_poly_list = []
        for i in inform:
            label_id = i["label"]
            points = i["points"]
            if "flag" == label_id.lower():  
                continue

            tmp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
            cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
            tmp_area = tmp_mask.sum()

            area_list.append(tmp_area)
            valid_poly_list.append(i)


        sort_index = np.argsort(area_list)[::-1].astype(np.int32)
        sort_index = list(sort_index)
        sort_inform = []
        for s_idx in sort_index:
            sort_inform.append(valid_poly_list[s_idx])

        mask = np.zeros((height, width), dtype=np.uint8)
        for i in sort_inform:
            label_id = i["label"]
            points = i["points"]

            if "ignore" in label_id.lower():
                label_value = 255 
            else:
                label_value = 1 

            cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

        return mask, comments, is_sentence

    def __getitem__(self, idx):


        image_path = self.images[idx]

        json_path = image_path.replace(".jpg", ".json")

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask, comments, is_sentence = self.get_mask_from_json(json_path, img)
        return image_path, mask, comments
