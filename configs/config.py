class ModelConfig:
    """Model configuration"""
    SAM_CHECKPOINT = " "
    OWL_MODEL_PATH= ''
    MODEL_TYPE = "vit_h"
    DEVICE = "cuda"

class PathConfig:
    """Path configuration"""
    OUTPUT_DIR_REASONSEG_TEST = ""
    OUTPUT_DIR_REASONSEG_VAL = ""
    OUTPUT_DIR_REASONSEG_TRAIN = ""
    OUTPUT_DIR_LLMSEG_VAL = ""
    OUTPUT_DIR_LLMSEG_TEST = ""
    OUTPUT_DIR_REFCOCO = ""
    OUTPUT_DIR_REFCOCOP = ""
    OUTPUT_DIR_REFCOCOG = ""
    

class APIConfig:
    """API configuration"""
    OPENAI_API_KEY = ''
    OPENAI_BASE_URL = ''
    GPT_MODEL = " "

class PromptConfig:
    """Prompt configuration"""
    GET_LEVEL_PROMPT = 'Now you need to complete a level classification task with two levels: 0 and 1. ' \
    '- Level 0 represents the object level, which means you should determine whether the segmentation answer is the whole object or multiple objects, not a part of an object. ' \
    '- Level 1 represents the part level, which means you should determine whether the segmentation answer is a part of an object, not the whole object.' \
    'Simply respond with the number 0 or 1. ' \
    'The following is the question and image.'
    GET_SEMANTIC_PROMPT = '### Task Description: You will be presented with two images. \
    The first image is a cropped section from the second image. Your task is to assign an accurate semantic label to the first image. \
    To complete this task, refer to the following three aspects: \
    [1]Semantic Information of the First Image: Analyze the image itself, identifying the object and its features. \
    [2]Relationship Between the First and Second Images: Based on the overall content of the second image, understand the relationship between the first image and the second image, particularly their position within the structure. \
    [3]Inquiry Content: If the first image provides the answer to the inquiry, incorporate the inquiry content directly into the semantic label. \
    ### Finally, provide a summary sentence in the format "This is XXXXX," describing only the first image. \
    ### Inquiry: '

