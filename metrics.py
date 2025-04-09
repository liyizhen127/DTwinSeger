import torch
import numpy as np

class IoUCalculator:
    """Calculator for IoU metrics including cIoU and gIoU"""
    def __init__(self, num_classes=2):
        """
        Initialize IoU calculator
        Args:
            num_classes: Number of classes (default: 2 for binary segmentation)
        """
        self.num_classes = num_classes
        # Accumulate intersection and union for all images
        self.total_intersection = torch.zeros(num_classes)
        self.total_union = torch.zeros(num_classes)
        # Store IoU of each image for gIoU calculation
        self.per_image_ious = []
        
    def reset(self):
        """Reset calculator state"""
        self.total_intersection = torch.zeros(self.num_classes)
        self.total_union = torch.zeros(self.num_classes)
        self.per_image_ious = []

    @staticmethod
    def calculateIouPerClass(predMask, gtMask, numClasses, ignoreIndex):
        """
        Calculate IoU for each class for a single image
        Args:
            predMask: Predicted mask (H, W)
            gtMask: Ground truth mask (H, W)
            numClasses: Number of classes
            ignoreIndex: Index to ignore in calculation
        Returns:
            Tuple: (intersection, union) tensors for each class
        """
        predMask = torch.from_numpy(predMask) if isinstance(predMask, np.ndarray) else predMask
        gtMask = torch.from_numpy(gtMask) if isinstance(gtMask, np.ndarray) else gtMask
        
        intersection = torch.zeros(numClasses, dtype=torch.float32)
        union = torch.zeros(numClasses, dtype=torch.float32)
        
        for cls in range(numClasses):
            predCls = (predMask == cls)
            gtCls = (gtMask == cls)
            validMask = (gtMask != ignoreIndex)
            predCls = predCls & validMask
            gtCls = gtCls & validMask
            intersection[cls] = (predCls & gtCls).sum().float()
            union[cls] = (predCls | gtCls).sum().float()
            
        return intersection, union

    def update(self, predMask, gtMask, ignoreIndex=255):
        """
        Update statistics with a single image
        Args:
            predMask: Predicted mask (H, W)
            gtMask: Ground truth mask (H, W)
            ignoreIndex: Index to ignore in calculation
        """
        assert predMask.shape == gtMask.shape, "Prediction and ground truth dimensions must match!"
        
        # Calculate intersection and union for current image
        intersection, union = self.calculateIouPerClass(
            predMask,
            gtMask,
            self.num_classes,
            ignoreIndex
        )
        
        # Update accumulated values
        self.total_intersection += intersection
        self.total_union += union
        
        # Calculate and store IoU for current image
        iou = intersection / (union + 1e-10)
        self.per_image_ious.append(iou[1] if self.num_classes > 1 else iou[0])

    def compute(self):
        """
        Compute accumulated cIoU and gIoU
        Returns:
            tuple: (cIoU, gIoU) for the target class
            - cIoU: cumulative IoU (sum of intersections / sum of unions)
            - gIoU: global IoU (mean of per-image IoUs)
        """
        if not self.per_image_ious:
            return 0.0, 0.0
            
        # Calculate cIoU using accumulated intersections and unions
        target_class = 1 if self.num_classes > 1 else 0
        ciou = self.total_intersection[target_class] / (self.total_union[target_class] + 1e-10)
        
        # Calculate gIoU as mean of per-image IoUs
        giou = torch.mean(torch.stack(self.per_image_ious))
        
        return ciou.item(), giou.item()