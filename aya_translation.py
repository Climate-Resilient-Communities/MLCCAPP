from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
import os

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Authenticate with Hugging Face API
login(token='enter your HF token here')

class LanguageSupport(Enum):
    COMMAND_R_PLUS = "command_r_plus"
    AYA = "aya"
    UNSUPPORTED = "unsupported"

MAX_NEW_TOKENS = 1024  # Define constants for magic numbers

def get_translation_prompt(input_language: str, output_language: str, text: str) -> str:
    """Return a prompt string formatted for translation purposes."""
    return f"Translate from {input_language} to {output_language}: '{text}'"

def get_message_format(prompts: List[str]) -> List[Dict[str, Any]]:
    """Return a message format for the Aya model input."""
    return [{"role": "user", "content": prompt} for prompt in prompts]

def generate_aya_23(
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    temperature: float = 0.3,
    top_p: float = 0.75,
    top_k: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> List[str]:
    """
    Generate text responses from the Aya model based on provided prompts.
    Raises ValueError if prompts list is empty.
    """
    if not prompts:
        raise ValueError("Prompts list cannot be empty.")
        
    messages = get_message_format(prompts)
    
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        padding=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])

    try:
        gen_tokens = model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]
        return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        raise

def aya_translation(
    input_language: str, output_language: str, text: str, 
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> str:
    """Translate text using the Aya model."""
    try:
        prompt = get_translation_prompt(input_language, output_language, text)
        prompts = [prompt]
        generations = generate_aya_23(prompts, model, tokenizer)
        return generations[0]
    except Exception as e:
        logging.error(f"Error during translation: {str(e)}")
        raise

class MultilingualRouter:
    COMMAND_R_PLUS_SUPPORTED_LANGUAGES = {
        'en', 'fr', 'es', 'it', 'de', 'pt-BR', 'ja', 'ko', 'zh-CN', 'ar',
        'ru', 'pl', 'tr', 'vi', 'nl', 'cs', 'id', 'uk', 'ro', 'el', 'hi', 'he', 'fa'
    }
    
    AYA_SUPPORTED_LANGUAGES = {
        'af', 'am', 'ar', 'az', 'be', 'bn', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 
        'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 
        'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 
        'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 
        'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 
        'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 
        'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'ur', 
        'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu'
    }

    LANGUAGE_CODE_MAP = {
        'zh-cn': 'zh', 'zh-tw': 'zh', 'pt-br': 'pt', 'pt-pt': 'pt',
        'en-us': 'en', 'en-gb': 'en', 'fr-ca': 'fr', 'fr-fr': 'fr',
        'es-es': 'es', 'es-mx': 'es', 'es-ar': 'es', 'de-de': 'de',
        'de-at': 'de', 'de-ch': 'de', 'nl-nl': 'nl', 'nl-be': 'nl',
        'it-it': 'it', 'it-ch': 'it', 'sv-se': 'sv', 'sv-fi': 'sv',
        'no-no': 'no', 'da-dk': 'da', 'fi-fi': 'fi', 'he-il': 'he',
        'ar-sa': 'ar', 'ar-eg': 'ar', 'ru-ru': 'ru', 'pl-pl': 'pl',
        'ja-jp': 'ja', 'ko-kr': 'ko', 'vi-vn': 'vi', 'id-id': 'id',
        'ms-my': 'ms', 'th-th': 'th', 'tr-tr': 'tr', 'uk-ua': 'uk',
        'bg-bg': 'bg', 'cs-cz': 'cs', 'hu-hu': 'hu', 'ro-ro': 'ro',
        'sk-sk': 'sk', 'sl-si': 'sl'
    }

    def __init__(self, aya_model: Optional[AutoModelForCausalLM] = None, 
                 aya_tokenizer: Optional[AutoTokenizer] = None):
        """Initialize router with Aya model and tokenizer if provided."""
        self.aya_model = aya_model
        self.aya_tokenizer = aya_tokenizer
        self.logger = logging.getLogger(__name__)

    def standardize_language_code(self, language_code: str) -> str:
        """Standardize language codes to match our supported formats."""
        return self.LANGUAGE_CODE_MAP.get(language_code.lower(), language_code.lower())

    def check_language_support(self, language_code: str) -> LanguageSupport:
        """Check the level of language support using standardized language codes."""
        std_lang_code = self.standardize_language_code(language_code)
        if std_lang_code in self.COMMAND_R_PLUS_SUPPORTED_LANGUAGES:
            return LanguageSupport.COMMAND_R_PLUS
        elif std_lang_code in self.AYA_SUPPORTED_LANGUAGES:
            return LanguageSupport.AYA
        return LanguageSupport.UNSUPPORTED

    def _get_unsupported_language_message(self, language_name: str, language_code: str) -> str:
        """Private helper to generate unsupported language messages."""
        return f"We currently don't support {language_name} ({language_code}). Please try another language."

    def translate_with_aya(self, query: str, language_code: str) -> str:
        """Translate the query using Aya model, logging errors if translation fails."""
        try:
            return aya_translation(
                input_language=language_code,
                output_language="en",
                text=query,
                model=self.aya_model,
                tokenizer=self.aya_tokenizer
            )
        except Exception as e:
            self.logger.error(f"Translation failed for language '{language_code}': {e}")
            raise

    def route_query(self, query: str, language_code: str, language_name: str) -> Dict[str, Any]:
        """Route the query based on standardized language code and language support level."""
        standardized_code = self.standardize_language_code(language_code)
        support_level = self.check_language_support(standardized_code)
        
        routing_result = {
            'processed_query': query,
            'original_language': standardized_code,
            'should_proceed': support_level != LanguageSupport.UNSUPPORTED,
            'routing_info': {
                'support_level': support_level.value,
                'needs_translation': False,
                'message': None
            }
        }
        
        if support_level == LanguageSupport.UNSUPPORTED:
            routing_result['routing_info']['message'] = self._get_unsupported_language_message(language_name, standardized_code)
            return routing_result

        if support_level == LanguageSupport.AYA:
            translated_query = self.translate_with_aya(query, standardized_code)
            routing_result.update({
                'processed_query': translated_query,
                'routing_info': {
                    'support_level': support_level.value,
                    'needs_translation': True,
                    'message': "Translated with Aya for further processing"
                }
            })

        return routing_result

def initialize_multilingual_pipeline(model_name: str = "CohereForAI/aya-23-8b") -> MultilingualRouter:
    """Initialize the multilingual pipeline with required models."""
    try:
        aya_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        aya_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return MultilingualRouter(aya_model, aya_tokenizer)
    except Exception as e:
        logging.error(f"Failed to initialize multilingual pipeline: {str(e)}")
        raise

####code to test the multilingual pipeline
# if __name__ == "__main__":
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Initialize the router
#     router = initialize_multilingual_pipeline()
    
#     # Example queries in different languages with user-provided language codes and names
#     test_queries = [
#         ("What is climate change?", "en", "English"),
#         ("¿Qué es el cambio climático?", "es", "Spanish"),
#         ("气候变化是什么？", "zh", "Chinese"),
#         ("जलवायु परिवर्तन क्या है?", "hi", "Hindi"),
#         ("Klimatske promjene", "bs", "Bosnian"),
#         ("Wat is klimaatverandering?", "nl", "Dutch")
#     ]
    
#     for query, lang_code, language in test_queries:
#         print(f"\nTesting {language} query: {query}")
#         result = router.route_query(query, lang_code, language)
        
#         print(f"Language code: {result['original_language']}")
#         print(f"Support level: {result['routing_info']['support_level']}")
#         print(f"Should proceed: {result['should_proceed']}")
#         print(f"Message: {result['routing_info']['message']}")
#         if result['routing_info'].get('needs_translation'):
#             print(f"Translated query: {result['processed_query']}")
#         print("-" * 50)