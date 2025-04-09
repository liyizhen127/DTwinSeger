import torch

def convertTensorsToSerializable(obj):
    """Convert tensor objects to JSON serializable format"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convertTensorsToSerializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: convertTensorsToSerializable(value) for key, value in obj.items()}
    return obj
