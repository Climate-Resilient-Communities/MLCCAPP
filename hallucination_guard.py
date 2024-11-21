from datasets import Dataset
import os
import sys
import asyncio
from ragas import evaluate
from dotenv import load_dotenv
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import FaithfulnesswithHHEM
from langchain_cohere import ChatCohere, CohereEmbeddings
from ragas.llms import LangchainLLMWrapper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def truncate_text(text: str, max_length: int = 450) -> str:
    """Truncate text to a maximum number of words while preserving meaning."""
    words = text.split()
    if len(words) <= max_length:
        return text
    return ' '.join(words[:max_length]) + '...'

def extract_contexts(citations, docs_reranked, max_contexts: int = 3):
    """
    Extract and truncate context from citations or documents.
    
    Args:
        citations: List of citation objects from Cohere
        docs_reranked: List of reranked documents
        max_contexts: Maximum number of contexts to include
        
    Returns:
        list: List of truncated context strings
    """
    contexts = []
    try:
        if citations is not None:
            for citation in citations[:max_contexts]:
                sources = citation.sources
                for source in sources:
                    document = source.document
                    context = f"{document.get('title', '')}: {document.get('snippet', '')}"
                    contexts.append(truncate_text(context))
        else:
            for document in docs_reranked[:max_contexts]:
                context = f"{document.get('title', '')}: {document.get('content', '')}"
                contexts.append(truncate_text(context))
        
        logger.debug(f"Extracted {len(contexts)} contexts")
        return contexts
        
    except Exception as e:
        logger.error(f"Error extracting contexts: {str(e)}")
        raise

async def check_hallucination(question: str, answer: str, contexts: list, cohere_api_key: str = None) -> float:
    """
    Check for hallucinations in the response.
    
    Args:
        question (str): Original question
        answer (str): Generated answer
        contexts (list): List of context strings
        cohere_api_key (str, optional): Cohere API key. If None, will try to get from environment
        
    Returns:
        float: Faithfulness score between 0 and 1
    """
    try:
        # Get API key if not provided
        if cohere_api_key is None:
            cohere_api_key = os.getenv('COHERE_API_KEY')
            if not cohere_api_key:
                raise ValueError("Cohere API key not found")
        
        # Initialize LLM wrapper
        evaluator_llm = LangchainLLMWrapper(
            ChatCohere(
                model="command-r-08-2024",
                cohere_api_key=cohere_api_key
            )
        )
        
        # Initialize scorer with LLM
        scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
        
        # Truncate question and answer if they're too long
        truncated_question = truncate_text(question, max_length=100)
        truncated_answer = truncate_text(answer, max_length=300)
        
        # Take only the most relevant contexts, truncated
        truncated_contexts = [truncate_text(ctx, max_length=200) for ctx in contexts[:3]]
        
        # Create evaluation sample
        input_sample = SingleTurnSample(
            user_input=truncated_question,
            response=truncated_answer,
            retrieved_contexts=truncated_contexts
        )
        
        # Get score
        score = await scorer.single_turn_ascore(input_sample)
        logger.info(f"Hallucination check complete. Score: {score}")
        return float(score)
        
    except Exception as e:
        logger.error(f"Error in hallucination check: {str(e)}")
        # Return a conservative score in case of error
        return 0.0

async def test_hallucination_guard():
    """Test function for hallucination checking with longer texts."""
    # Test data with longer texts
    test_cases = [
        {
            "question": "What are the effects of climate change?" * 5,  # Make it longer
            "answer": "Climate change leads to rising sea levels and increased global temperatures." * 10,
            "context": ["Climate change impacts: " + ("Rising sea levels, global temperature increase, and extreme weather events." * 20)],
            "expected_score_range": (0.7, 1.0)
        },
        {
            "question": "What are the effects of climate change?",
            "answer": "Climate change causes aliens to visit Earth more frequently." * 10,
            "context": ["Climate change impacts: " + ("Rising sea levels, global temperature increase, and extreme weather events." * 20)],
            "expected_score_range": (0.0, 0.3)
        }
    ]
    
    try:
        load_dotenv()
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("Cohere API key not found in environment")
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\nRunning test case {i + 1}")
            score = await check_hallucination(
                test_case["question"],
                test_case["answer"],
                test_case["context"],
                cohere_api_key
            )
            
            min_expected, max_expected = test_case["expected_score_range"]
            if min_expected <= score <= max_expected:
                logger.info(f"✓ Test case {i + 1} passed: Score {score} within expected range")
            else:
                logger.warning(f"✗ Test case {i + 1} failed: Score {score} outside expected range")
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run tests
        asyncio.run(test_hallucination_guard())
    else:
        # Run normal operation
        asyncio.run(main())