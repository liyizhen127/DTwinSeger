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
    GET_SEMANTIC_PROMPT = 'Now you need to complete a semantic assignment task. ' \
    'I will upload two images. The first image is extracted from the second image. ' \
    'You need to determine what the first image is based on the second image. '
    SUMMARY_PROMPT = 'Provide the best text for semantic assignment based on the above different content and original picture, as concise and clear as possible.'\
