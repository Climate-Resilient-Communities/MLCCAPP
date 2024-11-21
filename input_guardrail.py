import os
import ray
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

        print("Models initialized successfully")
        return topic_moderation_pipe
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

# if __name__ == "__main__":
    # # Initialize Ray and models
    # ray.init()
    # topic_moderation_pipe = initialize_models()
    
    # # Test cases
    # test_questions = [
    #     "What is climate change?",
    #     "How does global warming affect the environment?",
    #     "What is the capital of France?",
    #     "Tell me about climate adaptation strategies",
    #     "What are the climate impacts in Toronto?",
    #     "What are sustainable energy sources?"
    # ]
    
    # print("\nTesting Topic Moderation System")
    # print("=" * 60)
    
    # for question in test_questions:
    #     print(f"\nTesting Question: {question}")
        
    #     # Run topic check
    #     topic_result = topic_moderation.remote(question, topic_moderation_pipe)
        
    #     # Get and print result
    #     t_result = ray.get(topic_result)
    #     print(f"Topic Check: {t_result} (yes = climate-related, no = not climate-related)")
    #     print("=" * 60)