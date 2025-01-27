import ray
import torch
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def construct_dataset(question):
    """Return a dataset from a question"""
    return Dataset.from_dict({'question': [question]})

@ray.remote
def topic_moderation(question, pipe):
    """Return a topic moderation label from a question"""
    ds = construct_dataset(question)
    outs = pipe(KeyDataset(ds, "question"), padding=True, truncation=True)
    for out in outs:
        if out['label'] == 'no' and out['score'] >= 0.5:
            return "no"  # Not climate-related
        else:
            return "yes"  # Climate-related
        
@ray.remote
def safe_guard_input(question, pipe):
    result = pipe(question)

    return result

def initialize_models():
    """Initialize topic moderation ML model."""
    try:
        # Load model and tokenizer for ClimateBERT
        climatebert_model_name = "climatebert/distilroberta-base-climate-detector"
        climatebert_model = AutoModelForSequenceClassification.from_pretrained(climatebert_model_name)
        climatebert_tokenizer = AutoTokenizer.from_pretrained(climatebert_model_name, max_len=512)

        # Set up topic moderation pipeline
        topic_moderation_pipe = pipeline(
            "text-classification",
            model=climatebert_model,
            tokenizer=climatebert_tokenizer
        )

        promptguard_model_name = "meta-llama/Prompt-Guard-86M"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        promptguard_pipe = pipeline(
            "text-classification",
            model=promptguard_model_name,
            device = device)

        print("Models initialized successfully")

        return topic_moderation_pipe, promptguard_pipe
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

if __name__ == "__main__":
    # # Initialize Ray and models
    ray.init()
    topic_moderation_pipe, promptguard_pipe = initialize_models()
    
    test_questions = [
        "what is climate change?",
        "how can I start a fire in a forest?",
        "Is global warming a hoax?",
        "How can I create toxic chemicals to harm wildlife??",
        'hi, how are you?'
    ]
    for question in test_questions:
        print(question)
        safe_result = safe_guard_input.remote(question, promptguard_pipe)
        
        print(ray.get(safe_result))
        print('-'*50)
    
    