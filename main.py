import os
import ray
import sys
import time
import logging
import torch
import jsonlines
import cohere
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Dict, Any, Optional, List, Tuple
import asyncio

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel
from ragas.metrics import FaithfulnesswithHHEM
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset 
from langchain_cohere import ChatCohere, CohereEmbeddings

import warnings
warnings.filterwarnings("ignore", category=Warning)

load_dotenv('.env')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if COHERE_API_KEY:
    cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
else:
    raise EnvironmentError("COHERE_API_KEY not found in environment variables.")

# Import all functions from our custom modules
from aya_translation import (
    initialize_multilingual_pipeline,
    #MultilingualRouter,
    #LanguageSupport,
    aya_translation
)

from input_guardrail import (
    topic_moderation,
    # prompt_guard, #removing until we get prompt guard working
    #construct_dataset
    # get_class_probabilities,
    # get_jailbreak_score,
    # get_indirect_injection_score
)

from retrieval import (
    #get_query_embeddings,
    #weight_by_alpha,
    #issue_hybrid_query,
    get_hybrid_results,
    #get_entire_content,
    process_search_results #added this function to process search results
)

from rerank import rerank_fcn

from gen_response_and_citations import (
    #doc_preprocessing,
    cohere_chat
)

from hallucination_guard import (
    extract_contexts,
    check_hallucination,
    #FaithfulnesswithHHEM,
    #SingleTurnSample
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Define the correct index name as a constant
PINECONE_INDEX_NAME = "climate-change-adaptation-index-10-24-prod"

class MultilingualClimateChatbot:
    """
    A multilingual chatbot specialized in climate-related topics.
    
    This chatbot supports multiple languages through Aya translation,
    implements RAG (Retrieval Augmented Generation), and includes
    various guardrails for input validation and output quality.
    """

    # Language mappings from aya_translation.py
    LANGUAGE_NAME_TO_CODE = {
        'afrikaans': 'af', 'amharic': 'am', 'arabic': 'ar', 'azerbaijani': 'az',
        'belarusian': 'be', 'bengali': 'bn', 'bulgarian': 'bg', 'catalan': 'ca',
        'cebuano': 'ceb', 'czech': 'cs', 'welsh': 'cy', 'danish': 'da',
        'german': 'de', 'greek': 'el', 'english': 'en', 'esperanto': 'eo',
        'spanish': 'es', 'estonian': 'et', 'basque': 'eu', 'persian': 'fa',
        'finnish': 'fi', 'filipino': 'fil', 'french': 'fr', 'western frisian': 'fy',
        'irish': 'ga', 'scots gaelic': 'gd', 'galician': 'gl', 'gujarati': 'gu',
        'hausa': 'ha', 'hebrew': 'he', 'hindi': 'hi', 'croatian': 'hr',
        'hungarian': 'hu', 'armenian': 'hy', 'indonesian': 'id', 'igbo': 'ig',
        'icelandic': 'is', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jv',
        'georgian': 'ka', 'kazakh': 'kk', 'khmer': 'km', 'kannada': 'kn',
        'korean': 'ko', 'kurdish': 'ku', 'kyrgyz': 'ky', 'latin': 'la',
        'luxembourgish': 'lb', 'lao': 'lo', 'lithuanian': 'lt', 'latvian': 'lv',
        'malagasy': 'mg', 'macedonian': 'mk', 'malayalam': 'ml', 'mongolian': 'mn',
        'marathi': 'mr', 'malay': 'ms', 'maltese': 'mt', 'burmese': 'my',
        'nepali': 'ne', 'dutch': 'nl', 'norwegian': 'no', 'nyanja': 'ny',
        'odia': 'or', 'punjabi': 'pa', 'polish': 'pl', 'pashto': 'ps',
        'portuguese': 'pt', 'romanian': 'ro', 'russian': 'ru', 'sindhi': 'sd',
        'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'samoan': 'sm',
        'shona': 'sn', 'somali': 'so', 'albanian': 'sq', 'serbian': 'sr',
        'sesotho': 'st', 'sundanese': 'su', 'swedish': 'sv', 'swahili': 'sw',
        'tamil': 'ta', 'telugu': 'te', 'tajik': 'tg', 'thai': 'th', 'turkish': 'tr',
        'ukrainian': 'uk', 'urdu': 'ur', 'uzbek': 'uz', 'vietnamese': 'vi',
        'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'chinese': 'zh',
        'zulu': 'zu'
    }

    LANGUAGE_VARIATIONS = {
        'mandarin': 'zh',
        'mandarin chinese': 'zh',
        'chinese mandarin': 'zh',
        'simplified chinese': 'zh',
        'traditional chinese': 'zh',
        'brazilian portuguese': 'pt',
        'portuguese brazilian': 'pt',
        'castilian': 'es',
        'castellano': 'es',
        'farsi': 'fa',
        'tagalog': 'fil',
        'standard chinese': 'zh'
    }
    def __init__(self, index_name: str, input_jsonl_path: str):
        """Initialize the chatbot with necessary components."""
        try:
            # Initialize environment
            env_path = ".env"
            if not os.path.exists(env_path):
                raise FileNotFoundError(f".env file not found at {env_path}")
            
            load_dotenv(env_path)
            self._initialize_api_keys()
            self._initialize_components(index_name, input_jsonl_path)
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_api_keys(self) -> None:
        """Initialize and validate API keys."""
        self.PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        self.COHERE_API_KEY = os.getenv('COHERE_API_KEY')
        self.TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
        
        if not all([self.PINECONE_API_KEY, self.COHERE_API_KEY, self.TAVILY_API_KEY]):
            raise ValueError("Missing required API keys in .env file")
            
        os.environ["COHERE_API_KEY"] = self.COHERE_API_KEY
        os.environ["TAVILY_API_KEY"] = self.TAVILY_API_KEY

    def _initialize_components(self, index_name: str, input_jsonl_path: str) -> None:
        """Initialize all required components."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Load documents
        if not os.path.exists(input_jsonl_path):
            raise FileNotFoundError(f"Input JSONL file not found: {input_jsonl_path}")
            
        self.original_docs = []
        with jsonlines.open(input_jsonl_path) as reader:
            for obj in reader:
                self.original_docs.append(obj)
        
        # Debug logging for document structure
        logger.info(f"Loaded {len(self.original_docs)} original documents")
        if self.original_docs:
            sample_doc = self.original_docs[0]
            logger.info(f"Sample document structure: {list(sample_doc.keys())}")
            if 'metadata' in sample_doc:
                logger.info(f"Sample metadata structure: {type(sample_doc['metadata'])}")
                if isinstance(sample_doc['metadata'], list) and sample_doc['metadata']:
                    logger.info(f"First metadata item structure: {list(sample_doc['metadata'][0].keys())}")
        
        # Initialize components
        self._initialize_models()
        self._initialize_retrieval(index_name)
        self._initialize_language_router()
        
        # Initialize storage
        self.response_cache = {}
        self.conversation_history = []
        self.feedback_metrics = []

    def _initialize_models(self) -> None:
        """Initialize all ML models."""
        # Initialize ClimateBERT for topic moderation
        self.climatebert_model = AutoModelForSequenceClassification.from_pretrained(
            "climatebert/distilroberta-base-climate-detector"
        )
        self.climatebert_tokenizer = AutoTokenizer.from_pretrained(
            "climatebert/distilroberta-base-climate-detector",
            max_len=512
        )
        
        # Initialize Prompt Guard
        #self.promptguard_model = AutoModelForSequenceClassification.from_pretrained(
        #    "meta-llama/Prompt-Guard-86M"
        #)
        #self.promptguard_tokenizer = AutoTokenizer.from_pretrained(
        #    "meta-llama/Prompt-Guard-86M"
        #)

        self.promptguard_model = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceOffline/promptguard"
        )
        self.promptguard_tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceOffline/promptguard"
        )
        
        # Set up pipelines
        device = 0 if torch.cuda.is_available() else -1 
        self.topic_moderation_pipe = pipeline(
            "text-classification",
            model=self.climatebert_model,
            tokenizer=self.climatebert_tokenizer,
            device=device,
            truncation=True,
            max_length=512
        )

    def _initialize_retrieval(self, index_name: str) -> None:
        """Initialize retrieval components."""
        self.pinecone_client = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = self.pinecone_client.Index(index_name)
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

    def _initialize_language_router(self) -> None:
        """Initialize language routing components."""
        self.router = initialize_multilingual_pipeline()
        self.evaluator_llm = LangchainLLMWrapper(
            ChatCohere(model="command-r-08-2024")
        )
        self.embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0"
        )

    def get_language_code(self, language_name: str) -> str:
        """Convert language name to code."""
        language_name = language_name.lower().strip()
        
        if language_name in self.LANGUAGE_NAME_TO_CODE:
            return self.LANGUAGE_NAME_TO_CODE[language_name]
            
        if language_name in self.LANGUAGE_VARIATIONS:
            return self.LANGUAGE_VARIATIONS[language_name]
        
        available_languages = sorted(set(list(self.LANGUAGE_NAME_TO_CODE.keys()) + 
                                      list(self.LANGUAGE_VARIATIONS.keys())))
        raise ValueError(
            f"Unsupported language: {language_name}\n" +
            f"Available languages:\n" +
            f"{', '.join(available_languages)}"
        )
    async def process_input_guards(self, query: str) -> Dict[str, bool]:
        """Run input guardrails for topic moderation only."""
        try:
            logger.info("Running input guardrails")
            # Perform topic check
            topic_check = topic_moderation.remote(
                query,
                self.topic_moderation_pipe
            )
            
            # Get result
            topic_result = ray.get(topic_check)
            
            logger.debug(f"Guard results - Topic: {topic_result}")
            
            # Return only topic moderation result
            return {
                "passed": topic_result == "yes",
                "topic_check": topic_result == "yes"
            }
            
        except Exception as e:
            logger.error(f"Error in input guards: {str(e)}")
            raise

    async def get_documents(self, query: str) -> List[Dict]:
        """Retrieve and rerank relevant documents."""
        try:
            logger.info("Starting document retrieval")
            
            # Get hybrid search results
            try:
                hybrid_results = get_hybrid_results(
                    self.index,
                    query,
                    self.embed_model,
                    alpha=0.4,
                    top_k=10
                )

                logger.debug(f"Retrieved {len(hybrid_results.matches)} matches from hybrid search")
                
                # Debug first result structure
                if hybrid_results.matches:
                    logger.debug(f"First match metadata: {hybrid_results.matches[0].metadata}")
                
            except Exception as e:
                logger.error(f"Error in hybrid search: {str(e)}")
                raise
                
            # Process results - this handles content extraction and cleaning
            try:
                docs_with_content = process_search_results(hybrid_results)
                logger.debug(f"Processed {len(docs_with_content)} documents")
                
                # Debug processed documents
                if docs_with_content:
                    logger.debug(f"First processed doc structure: {docs_with_content[0].keys()}")
                    logger.debug(f"First processed doc title: {docs_with_content[0].get('title', 'No title')}")
                    logger.debug(f"First processed doc content length: {len(docs_with_content[0].get('content', ''))}")
                
                if not docs_with_content:
                    logger.warning("No documents with content found for reranking")
                    return []
                    
            except Exception as e:
                logger.error(f"Error in processing search results: {str(e)}")
                raise
            
            # Rerank documents
            try:
                reranked_docs = rerank_fcn(
                    query=query,
                    docs_to_rerank=docs_with_content,
                    top_k=5
                )
                
                logger.info(f"Successfully reranked {len(reranked_docs)} documents")
                
                # Debug reranked documents
                if reranked_docs:
                    logger.debug(f"First reranked doc structure: {reranked_docs[0].keys()}")
                    logger.debug(f"First reranked doc title: {reranked_docs[0].get('title', 'No title')}")
                    logger.debug(f"First reranked doc content length: {len(reranked_docs[0].get('content', ''))}")
                else:
                    logger.warning("No documents after reranking")
                
                return reranked_docs
                
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}", exc_info=True)
            raise

    async def process_query(
            self,
            query: str,
            language_name: str
        ) -> Dict[str, Any]:
            """Process a query through the complete pipeline."""
            try:
                start_time = time.time()
                logger.info("🔍 Starting search process...")
                
                # 0. Check cache first
                language_code = self.get_language_code(language_name)
                cache_key = f"{language_code}:{query.lower().strip()}"
                
                if cache_key in self.response_cache:
                    logger.info("📚 Found cached response...")
                    cached_result = self.response_cache[cache_key]
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    
                    processing_time = time.time() - start_time
                    return {
                        "success": True,
                        "response": cached_result['response'],
                        "citations": cached_result['citations'],
                        "faithfulness_score": cached_result['faithfulness_score'],
                        "processing_time": processing_time,
                        "cache_hit": True
                    }
                
                # 1. Language routing
                logger.info("🌐 Processing language routing...")
                language_code = self.get_language_code(language_name)
                
                route_result = self.router.route_query(
                    query=query,
                    language_code=language_code,
                    language_name=language_name
                )
                
                if not route_result['should_proceed']:
                    logger.warning(f"🌐 Language routing failed: {route_result['routing_info']['message']}")
                    return {
                        "success": False,
                        "message": route_result['routing_info']['message']
                    }
                
                processed_query = route_result['processed_query']
                english_query = route_result['english_query']
                logger.info("🌐 Language routing complete")
                
                # 2. Input validation
                logger.info("🔍 Validating input...")
                guard_results = await self.process_input_guards(english_query)

                if not guard_results['passed']:
                    logger.warning("🔍 Query failed input validation")
                    return {
                        "success": False,
                        "message": "Query failed input validation checks"
                    }
                logger.info("🔍 Input validation passed")
                
                # 3. Document retrieval and reranking (all in one step)
                try:
                    logger.info("📚 Starting retrieval and reranking...")
                    reranked_docs = await self.get_documents(processed_query)
                    logger.info(f"📚 Reranked {len(reranked_docs)} documents")
                    
                except Exception as e:
                    logger.error(f"📚 Error in retrieval process: {str(e)}")
                    raise
                
                # 4. Generate response - directly use reranked_docs
                try:
                    logger.info("✍️ Starting response generation...")
                    response, citations = cohere_chat(processed_query, reranked_docs)
                    logger.info("✍️ Response generation complete")
                    
                except Exception as e:
                    logger.error(f"✍️ Error in response generation: {str(e)}")
                    raise

                # 5. Quality checks
                logger.info("✔️ Starting quality checks...")
                try:
                    # Extract contexts with limits
                    logger.info("✔️ Extracting contexts for verification...")
                    contexts = extract_contexts(citations, reranked_docs, max_contexts=3)

                    # Check for hallucinations
                    logger.info("✔️ Performing hallucination check...")

                    # Translate the response to English if it is not already; then check for hallucination.
                    if route_result['routing_info']['support_level']=='command_r_plus' and language_code!='en':
                        processed_response = self.router.translate_with_command_r(response, language_name)
                    else:
                        processed_response = response

                    faithfulness_score = await check_hallucination(
                        question=english_query,
                        answer=processed_response,
                        contexts=contexts,
                        cohere_api_key=self.COHERE_API_KEY
                    )
                    logger.info(f"✔️ Hallucination check complete - Score: {faithfulness_score}")
                    
                    if faithfulness_score < 0.7:
                        logger.warning("✔️ Low faithfulness score detected, attempting recovery...")
                        
                        # First try: Regenerate with stricter prompt
                        logger.info("✔️ Attempting regeneration with stricter prompt...")
                        description = 'Please ensure strict factual accuracy'
                        regenerated_response, regenerated_citations = cohere_chat(processed_query, reranked_docs, description)

                        # Check regenerated response
                        logger.info("✔️ Checking regenerated response...")
                        new_contexts = extract_contexts(regenerated_citations, reranked_docs, max_contexts=3)

                        # Translate the response to English if it is not already; then check for hallucination.
                        if route_result['routing_info']['support_level']=='command_r_plus' and language_code!='en':
                            processed_response = self.router.translate_with_command_r(regenerated_response, language_name)
                        else:
                            processed_response = regenerated_response

                        regenerated_score = await check_hallucination(
                            question=english_query,
                            answer=processed_response,
                            contexts=new_contexts,
                            cohere_api_key=self.COHERE_API_KEY
                        )
                        logger.info(f"✔️ Regenerated response score: {regenerated_score}")
                        
                        # If regeneration didn't help, try Tavily fallback
                        if regenerated_score < 0.7:
                            logger.info("✔️ Attempting Tavily fallback...")
                            fallback_response, fallback_citations, fallback_score = await self._try_tavily_fallback(
                                query=processed_query,
                                english_query = english_query,
                                language_name=language_name
                            )
                            
                            # Use fallback results if they're better
                            if fallback_response and fallback_score > faithfulness_score:
                                logger.info("✔️ Using Tavily fallback response")
                                response = fallback_response
                                citations = fallback_citations
                                faithfulness_score = fallback_score
                            else:
                                # If fallback didn't help, use regenerated response if it was better
                                if regenerated_score > faithfulness_score:
                                    logger.info("✔️ Using regenerated response")
                                    response = regenerated_response
                                    citations = regenerated_citations
                                    faithfulness_score = regenerated_score
                                else:
                                    # Otherwise, stick with original response but add a warning
                                    logger.info("✔️ Using original response with warning")
                                    response += "\n\nNote: This response may contain uncertainties. Please verify critical information from additional sources."
                        else:
                            # Use regenerated response if it's better
                            logger.info("✔️ Using improved regenerated response")
                            response = regenerated_response
                            citations = regenerated_citations
                            faithfulness_score = regenerated_score
                            
                except Exception as e:
                    logger.error(f"✔️ Error in quality checks: {str(e)}")
                    faithfulness_score = 0.0  # Conservative score on error

                # 6. Language processing
                if route_result['routing_info']['needs_translation']:
                    logger.info(f"🌐 Translating response back to {language_name}")
                    response = aya_translation(
                        "en",
                        language_code,
                        response,
                        self.router.aya_model,
                        self.router.aya_tokenizer
                    )
                    logger.info("🌐 Translation complete")
                
                # Store results and return
                processing_time = time.time() - start_time
                logger.info("✨ Storing results...")
                await self._store_results(
                    query=query,
                    response=response,
                    language_code=language_code,
                    citations=citations,
                    faithfulness_score=faithfulness_score,
                    processing_time=processing_time,
                    route_result=route_result
                )
                
                logger.info("✨ Processing complete!")
                return {
                    "success": True,
                    "response": response,
                    "citations": citations,
                    "faithfulness_score": faithfulness_score,
                    "processing_time": processing_time,
                    "cache_hit": False
                }
                    
            except Exception as e:
                logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "message": f"Error processing query: {str(e)}"
                }
            
    async def _try_tavily_fallback(self, query: str, english_query: str, language_name: str) -> Tuple[Optional[str], Optional[List], float]:
        """
        Attempt to get a response using Tavily search when primary response fails verification.
        
        Args:
            query: The original query
            english_query: Translated query
            language_name: Original language
            
        Returns:
            Tuple of (response text, citations, faithfulness score)
        """
        try:
            logger.info("Attempting Tavily fallback search")
            tavily_search = TavilySearchResults()

            # Perform web search
            search_results = await tavily_search.ainvoke(query)
            
            if not search_results:
                logger.warning("No results from Tavily search")
                return None, None, 0.0
                
            # Format documents for Cohere
            documents_for_cohere = []
            for result in search_results:
                document = {
                        'title': result.get('url', ''),
                        'url': result.get('url', ''),
                        'content': result.get('content', '')
                    }
                documents_for_cohere.append(document)
            
            # Generate new response with Tavily results
            description = """Please provide accurate information based on the search results. Always cite your sources. Ensure strict factual accuracy"""
            fallback_response, fallback_citations = cohere_chat(query, documents_for_cohere, description)
            
            # Verify fallback response
            web_contexts = [f"{result.get('title', '')}: {result.get('content', '')}" 
                        for result in search_results]
            
            # Translate the fallback response and web context to English if they are not already; then check for hallucination.
            if query != english_query:
                processed_response = self.router.translate_with_command_r(fallback_response, language_name)
                processed_context = self.router.translate_with_command_r(web_contexts, language_name)
            else:
                processed_response = fallback_response
                processed_context = web_contexts

            # Combine original and web contexts for better verification
            #combined_contexts = original_contexts + web_contexts
            
            # Check faithfulness of fallback response
            fallback_score = await check_hallucination(
                question=english_query,
                answer=processed_response,
                contexts=processed_context,  # Limit contexts to prevent overload
                cohere_api_key=self.COHERE_API_KEY
            )
            
            return fallback_response, fallback_citations, fallback_score
            
        except Exception as e:
            logger.error(f"Error in Tavily fallback: {str(e)}")
            return None, None, 0.0
        
    async def _store_results(
        self,
        query: str,
        response: str,
        language_code: str,
        citations: List[Any],
        faithfulness_score: float,
        processing_time: float,
        route_result: Dict[str, Any]
    ) -> None:
        """
        Store query results in the chatbot's cache and update metrics.
        
        Args:
            query (str): Original query
            response (str): Generated response
            language_code (str): Language code of the query
            citations (List[Any]): List of citations used
            faithfulness_score (float): Hallucination check score
            processing_time (float): Time taken to process query
            route_result (Dict[str, Any]): Language routing information
        """
        try:
            # Cache response - simplified without timestamp
            cache_key = f"{language_code}:{query.lower().strip()}"
            self.response_cache[cache_key] = {
                "response": response,
                "citations": citations,
                "faithfulness_score": faithfulness_score
            }
            
            # Update conversation history (kept for user experience)
            self.conversation_history.append({
                "query": query,
                "response": response,
                "language": language_code,
                "faithfulness_score": faithfulness_score
            })
            
            # Store metrics (kept for monitoring/analytics)
            self.feedback_metrics.append({
                "language": language_code,
                "processing_time": processing_time,
                "required_translation": route_result['routing_info']['needs_translation'],
                "faithfulness_score": faithfulness_score
            })
            
            logger.debug(f"Results stored successfully for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
            # Don't raise the error as this is a non-critical operation

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if ray.is_initialized():
                ray.shutdown()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Main entry point for the climate chatbot application."""
    try:
        # Validate command line arguments
        if len(sys.argv) < 3:
            print("Usage: python main.py <index_name> <input_jsonl_path>")
            print("Example: python main.py climate-change-adaptation-index-10-24-prod ./data/climate_docs.jsonl")
            sys.exit(1)
            
        index_name = sys.argv[1]
        input_jsonl_path = sys.argv[2]

        # Initialize chatbot
        print("\nInitializing Climate Chatbot...")
        chatbot = MultilingualClimateChatbot(index_name, input_jsonl_path)
        print("✓ Initialization complete\n")
        
        # Print welcome message
        print("Welcome to the Multilingual Climate Chatbot!")
        print("Available languages:")
        languages = sorted(set(list(chatbot.LANGUAGE_NAME_TO_CODE.keys()) + 
                             list(chatbot.LANGUAGE_VARIATIONS.keys())))
        
        # Print languages in columns
        col_width = 20
        num_cols = 4
        for i in range(0, len(languages), num_cols):  
            row = languages[i:i + num_cols]
            print("".join(lang.ljust(col_width) for lang in row))
            
            # Get language choice once at the start
        while True:
            language_name = input("\nPlease select your language for this session: ").strip()
            if language_name:
                try:
                    # Validate language selection
                    chatbot.get_language_code(language_name)
                    print(f"\nLanguage set to: {language_name}")
                    break
                except ValueError as e:
                    print(f"\nError: {str(e)}")
                    continue
        
        print("\nType 'quit' to exit, 'language' to see your current language setting\n")

        # Main interaction loop
        while True:
            try:
                # Get query
                query = input("\nEnter your question: ").strip()
                if not query:
                    print("Please enter a question.")
                    continue
                    
                if query.lower() == 'quit':
                    print("\nThank you for using the Climate Chatbot!")
                    break
                    
                if query.lower() == 'languages':
                    print(f"\nCurrent language: {language_name}")
                    continue
                
                # # Get language
                # language_name = input("Enter language name: ").strip()
                # if not language_name:
                #     print("Please enter a language name.")
                #     continue
                
                print("\nProcessing your query...")
                
                # Process query
                result = await chatbot.process_query(
                    query=query,
                    language_name=language_name
                )
                
                # Display results
                if result['success']:
                    print("\nResponse:", result['response'])
                    
                    if result['citations']:
                        print("\nSources:")
                        for citation in result['citations']:
                            print(f"- {citation}")
                            
                    print(f"\nFaithfulness Score: {result['faithfulness_score']:.2f}")
                    print(f"Processing Time: {result['processing_time']:.2f} seconds")
                else:
                    print("\nError:", result['message'])
                    
                print("\n" + "-"*50)  # Separator line
                    
            except KeyboardInterrupt:
                print("\n\nExiting gracefully...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")
                
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        raise
    finally:
        if 'chatbot' in locals():
            print("\nCleaning up resources...")
            chatbot.cleanup()
            print("✓ Cleanup complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nProgram terminated due to error: {str(e)}")
        sys.exit(1)