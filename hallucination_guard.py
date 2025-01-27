import os
import sys
import asyncio
from utils.env_loader import load_environment
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM
from langchain_cohere import ChatCohere
from ragas.llms import LangchainLLMWrapper
import logging
import warnings
from transformers import logging as transformers_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
transformers_logging.set_verbosity_error()  # This will suppress model loading warnings

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
            "question": "how does climate change affect human life?",  # Make it longer
            "answer": """
                        Climate change is a big deal for all of us, and it's affecting our lives in lots of ways. Here's a look at some of the impacts:
                        - **Health:**
                            - **Physical Health:** Climate change can cause all sorts of health problems. For example, extreme weather like heatwaves, floods, and storms can make people sick or even hurt them. It can also mess with the air we breathe and the water we drink, which isn't good for our bodies.
                            - **Mental Health:** Some people might feel worried or anxious about climate change. This is called "eco-anxiety," and it's a real thing.
                        - **Environment:**
                            - **Nature:** Our natural world is changing because of climate change. This can mean losing plants and animals, and even whole ecosystems.
                            - **Extreme Events:** Wildfires, droughts, and big storms can damage our homes and the places we live.
                        - **Daily Life:**
                            - **Food and Water:** Climate change can make it harder to grow food and find clean water.
                            - **Work and Money:** Some jobs and businesses might be affected, especially if they depend on nature or the weather.

                        Remember, these are just a few examples. Climate change is a huge topic, and it touches almost every part of our lives. It's important to learn more and take action to protect our planet and ourselves!
                        """,
            "context": ["""
                        Integrating Social and Behavior Change in Climate Change Adaptation- An Introductory Guide: https://www.climatelinks.org/sites/default/files/asset/document/2019_USAID_ATLAS_SBC%20Guide.pdf: 
                        Climate variability and change will affect the health and livelihoods of most populations in the coming years, putting the lives and well-being of millions of people at increased risk. 
                        Rising temperatures threaten many crop species, livestock health, and agricultural systems. 
                        Rainfed agriculture is particularly vulnerable to rising temperatures, a potential increase in extreme weather events, and changes in precipitation patterns. 
                        Sea level rise, more intense storm surge, and ocean warming and acidification all have potentially devastating impacts on human life and natural, social, and physical assets in coastal areas. 
                        Public health is directly impacted by threats such as heat waves, and indirectly impacted by changes in the distribution and transmission of diseases and heightened food insecurity. 
                        The poorest countries and communities are often the most vulnerable to these impacts because they lack the ability to prepare for and recover from both long-term changes, such as rising temperatures and sea level, and short-term shocks, such as more intense storms and floods. 
                        Significant adverse social, economic, and environmental impacts of climate change will arise as climate risks challenge traditional livelihoods, exacerbate conflicts, and intensify humanitarian crises. 
                        Most climate change adaptation and risk reduction measures require that humans modify existing behaviors or adopt new ones related to health, agriculture, natural resource management, infrastructure, and settlement patterns. 
                        SBC, which incorporates knowledge from across disciplines to change 1""",
                        """
                        National Adaptation Strategy for Canada - Canada.ca: https://www.canada.ca/en/services/environment/weather/climatechange/climate-plan/national-adaptation-strategy/full-strategy.html: 
                        Our natural environment is affected by climate change in both dramatic and subtle ways. 
                        A thriving natural environment is foundational for all forms of life, including people, our society and well-being. 
                        Widespread wildfires, frequent drought conditions, severe storms and permafrost thaw have a range of impacts on the environment, including loss of, and shifts in species, habitat and ecosystems. 
                        These have lasting impacts on the natural environment and add to the existing effects of pollution and habitat destruction. As humans, we exist as part of the natural environment, not in separation from it. 
                        Our relationship with the land is threatened with the degradation and loss of ecosystems that we rely on for our basic needs, like productive soil for food, access to clean air and water, or protection from floods. 
                        The environment also supports our recreational, health, and spiritual needs. Many of our cultural identities are closely tied to our connection with the natural environment, where the intrinsic value of nature shapes our sense of self and connection to the land. 
                        Extreme events, such as floods or wildfires, as well as slow-onset changes, such as thawing permafrost, extreme heat, or rising sealevels, can damage and destroy our infrastructure. 
                        Across Canada, we have seen the recent impacts that climate change can have on our homes and the infrastructure that we rely on everyday for essential services. 
                        Lost and damaged infrastructure endangers people's health and safety, while disrupting businesses, market access, and food, energy and trade supply chains. 
                        The damage caused by climate change creates major repair and rebuilding costs, which deepens existing inequities and impacts the well-being of our communities. 
                        At a Insured damages from Hurricane Fiona in 2022 are now estimated to be over $800 million. 
                        Local communities and businesses were severely impacted by storm, with critical infrastructure, including fishing wharves and small harbours damaged or destroyed. minimum, the impacts of climate change on infrastructure are making it harder for us to get around, work, or access food and water. 
                        At worst, we will experience loss of life, our homes or entire communities. 
                        Across Canada, climate change is affecting our livelihoods, impacting the work we do to earn an income as well as activities that are essential to everyday life, like growing, hunting, or gathering traditional foods. 
                        Disruptions to livelihoods affect our ability to secure the necessities of life and provide for ourselves and our families. 
                        Forestry, fisheries, agriculture, mining, energy, transportation, and tourism are some of the sectors facing greater climate change risks. 
                        Our supply chains are vulnerable to climate-related impacts, especially when close trading partners are affected. 
                        Impacts to economic and financial systems affect investments, insurance costs, incomes, and job..."""],
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
        load_environment()
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