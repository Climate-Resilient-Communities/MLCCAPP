from typing import Dict, Any
from enum import Enum
import logging
from utils.env_loader import load_environment
from nova_flow import BedrockModel

class LanguageSupport(Enum):
    COMMAND_R_PLUS = "command_r_plus"
    NOVA = "nova"
    UNSUPPORTED = "unsupported"

class MultilingualRouter:
    COMMAND_R_PLUS_SUPPORTED_LANGUAGES = {
        'en', 'fr', 'es', 'it', 'de', 'pt', 'ja', 'ko', 'zh', 'ar',
        'ru', 'pl', 'tr', 'vi', 'nl', 'cs', 'id', 'uk', 'ro', 'el', 'hi', 'he', 'fa'
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

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def standardize_language_code(self, language_code: str) -> str:
        """Standardize language codes to match our supported formats."""
        return self.LANGUAGE_CODE_MAP.get(language_code.lower(), language_code.lower())

    def check_language_support(self, language_code: str) -> LanguageSupport:
        """Check the level of language support using standardized language codes."""
        if language_code in self.COMMAND_R_PLUS_SUPPORTED_LANGUAGES:
            return LanguageSupport.COMMAND_R_PLUS
        else: 
            return LanguageSupport.NOVA

    def _get_unsupported_language_message(self, language_name: str, language_code: str) -> str:
        """Private helper to generate unsupported language messages."""
        return f"We currently don't support {language_name} ({language_code}). Please try another language."
        
    def route_query(self, query: str, language_code: str, language_name: str, translation) -> Dict[str, Any]:
        """Route the query based on standardized language code and language support level."""
        standardized_code = self.standardize_language_code(language_code)
        
        english_query = query if standardized_code=='en' else translation(query, language_name, 'english')
        proceed = True if english_query else False
        support_level = self.check_language_support(standardized_code) if proceed else LanguageSupport.UNSUPPORTED

        routing_result = {
            'processed_query': query,
            'english_query': english_query,
            'original_language': standardized_code,
            'should_proceed': proceed,
            'routing_info': {
                'support_level': support_level.value,
                'needs_translation': False,
                'message': None
            }
        }
        
        if support_level == LanguageSupport.NOVA:
            routing_result.update({
                'processed_query': english_query,
                'english_query': english_query,
                'routing_info': {
                    'support_level': support_level.value,
                    'needs_translation': True,
                    'message': "Translated with Amazon Nova Micro for further processing"
                }
            })

        elif support_level == LanguageSupport.UNSUPPORTED:
            routing_result['routing_info']['message'] = self._get_unsupported_language_message(language_name, standardized_code)
            return routing_result

        return routing_result

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load environment variables from env_loader file
    load_environment()

    # Initialize the router
    router = MultilingualRouter()

    # Initialize the BedrockModel class with your model ID and get the translation function
    nova_model = BedrockModel("amazon.nova-micro-v1:0")
    translation = nova_model.nova_translation
    
    # Example queries in different languages with user-provided language codes and names
    test_queries = [
        ("What is climate change?", "en", "english"),
        ("¿Qué es el cambio climático?", "es", "Spanish"),
        #("气候变化是什么？", "zh", "chinese"),
        #("जलवायु परिवर्तन क्या है?", "hi", "hindi"),
        #("Klimatske promjene", "bs", "bosnian"),
        #("Wat is klimaatverandering?", "nl", "dutch"),
        #("Wat is klimaatsverandering? Is dit ’n foefie? Beïnvloed dit menslike lewe?", "af", "afrikaans")
    ]
    
    for query, lang_code, language in test_queries:
        print(f"\nTesting {language} query: {query}")
        result = router.route_query(query, lang_code, language, translation)
        print(result)
        print("-" * 50)