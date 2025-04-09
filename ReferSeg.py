import torch
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import numpy as np
import random
from refer import REFER
class ReferDataset(Dataset):

    def __init__(self, data_root, dataset_name='refcoco', split='val', 
                 splitBy=None, num_samples=20, seed=42):
        self.seed = seed
        random.seed(seed)
        

        if splitBy is None:
            if dataset_name == 'refcocog':
                splitBy = 'umd'
            else:
                splitBy = 'unc'
        
     
        

        self.refer = REFER(data_root, dataset_name, splitBy)
        self.dataset_name = dataset_name
        self.split = split
        self.splitBy = splitBy
        

        ref_ids = self.refer.getRefIds(split=split)

        
        

        image_ids = list(set([self.refer.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        image_ids.sort()  
        
        if len(image_ids) < num_samples:
            self.selected_image_ids = image_ids
        else:

            self.selected_image_ids = image_ids[:num_samples]
        

        self.selected_refs = []
        for img_id in self.selected_image_ids:

            img_ref_ids = [ref_id for ref_id in ref_ids if self.refer.Refs[ref_id]['image_id'] == img_id]

            if img_ref_ids:

                ref = self.refer.loadRefs(img_ref_ids[0])[0]
                self.selected_refs.append(ref)
        

    
    def __len__(self):
    
        return len(self.selected_refs)
    
    def __getitem__(self, index):

        ref = self.selected_refs[index]
        

        img = self.refer.loadImgs(ref['image_id'])[0]
        image_path = osp.join(self.refer.IMAGE_DIR, img['file_name'])
        

        mask_info = self.refer.getMask(ref)
        mask = mask_info['mask']
        

        if len(ref['sentences']) > 0:
            query = ref['sentences'][0]['sent']
        else:
            query = ""
        
        return {
            'image_path': image_path,
            'mask': mask,
            'query': query,
            'ref_id': ref['ref_id'],
            'cat_id': ref['category_id'],
            'bbox': self.refer.getRefBox(ref['ref_id']),
            'category': self.refer.Cats[ref['category_id']]
        }

def get_refer_dataloader(data_root, dataset_name='refcoco', split='val', 
                         splitBy=None, num_samples=20, batch_size=1, 
                         num_workers=0, seed=42):


    dataset = ReferDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split=split,
        splitBy=splitBy,
        num_samples=num_samples,
        seed=seed
    )
    

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader