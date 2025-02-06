import os
import sys
import asyncio
import logging
import time
from typing import List
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM
from ragas.llms import LangchainLLMWrapper
from langchain_cohere import ChatCohere
from utils.env_loader import load_environment


# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="You are using a model of type HHEMv2Config")

def truncate_text(text: str, max_length: int = 450) -> str:
    """Truncate text to a maximum number of words while preserving meaning."""
    words = text.split()
    return ' '.join(words[:max_length]) + '...' if len(words) > max_length else text

def extract_contexts(docs_reranked: List[dict], max_contexts: int = 3) -> List[str]:
    """Extract and truncate context from documents."""
    try:
        contexts = [
            truncate_text(doc.get('content', ''))
            for doc in docs_reranked[:max_contexts]
        ]
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
                model="command-r-plus",
                cohere_api_key=cohere_api_key
            )
        )
        
        # Initialize scorer with LLM
        scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
        
        # Truncate question and answer if they're too long
        truncated_question = truncate_text(question, max_length=100)
        truncated_answer = truncate_text(answer, max_length=200)
        truncated_contexts = [truncate_text(ctx, max_length=200) for ctx in contexts[:1]]
        
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
    question = "waht is climate change?"
    context = [ """Climate change - A persistent, long-term change in the state of the climate, measured by changes in the mean state and/or its variability. Climate change may be due to natural internal processes, natural external forcings such as volcanic eruptions and modulations of the solar cycle, or to persistent anthropogenic changes in the composition of the atmosphere or in land use (IPCC, 2014).""",
                """Climate change: Climate change refers to a change in the state of the climate that can be identified (e.g. by using statistical tests) by changes in the mean and/or the variability of its properties, and that persists for an extended period, typically decades or longer. Climate change may be due to natural internal processes or external forcing factors, or to persistent anthropogenic changes in the composition of the atmosphere or in land use. Note that the United Nations Framework Convention on Climate Change (UNFCCC) defines climate change as "a change of climate which is attributed directly or indirectly to human activity that alters the composition of the global atmosphere and which is in addition to natural climate variability observed over comparable time periods. " The UNFCCC thus makes a distinction between climate change attributable to human activities altering the atmospheric composition, and climate variability attributable to natural causes. (From Impacts to Adaptation : Canada in a Changing Climate, 2007, http://adaptation.nrcan.gc.ca/assess/2007/ch11/index_e.php#R) Greenhouse effect: Greenhouse gases effectively absorb infrared radiation, emitted by the Earth's surface, by the atmosphere itself due to the same gases and by clouds. Atmospheric radiation is emitted to all sides, including downward to the Earth's surface. Thus, greenhouse gases trap heat within the surface-troposphere system. This is called the greenhouse effect. http://www.ipcc.ch/publications_and_data/ar4/wg3/en/annex1sglossary-e-i.html Greenhouse gas (GHG ): Gaseous constituents of the atmosphere, both natural and anthropogenic, that absorb and emit radiation at specific wavelengths within the spectrum of infrared radiation emitted by the Earth 's surface, by the atmosphere itself and by clouds. Water vapour (H2O), carbon dioxide (CO2), nitrous oxide (N2O), methane (CH4) and ozone (O3) are the primary greenhouse gases in the Earth's atmosphere. In addition, there are a number of entirely human-made greenhouse gases in the atmosphere, such as the halocarbons and other chlorine- and bromine-containing substances. (From Impacts to Adaptation : Canada in a Changing Climate, 2007, http://adaptation.nrcan.gc.ca/assess/2007/ch11/index_e.php#R) Mitigation: Initiatives and measures to reduce the vulnerability of natural and human systems against actual or expected climate change effects. (IPCC, 2008) Natural Systems: Includes the natural environment (forests, wetlands, wildlife, etc.) plus human activities to monitor, protect and enhance the natural environment and educate the public on the current and future state of our natural systems. http://www.peelregion.ca/planning/climatechange/reports/pdf/climate-chan-strat-bgr.pdf Public Health : Involves the combination of programs, services and policies that protect and promote the health of all. Public health is involved in the enhancement of the health status of the population; reduction of disparities in health status among individual/groups within that population; preparation for and response to outbreaks and emergencies; and enhancing the sustainability of the healthcare system. http://www.peelregion.ca/planning/climatechange/reports/pdf/climate-chan-strat-bgr.pdf Urban forest: The trees, forests and associated organisms that grow near buildings and in gardens, green spaces, parks and golf courses located in village, town, suburban and urban areas. http://canadaforests.nrcan.gc.ca/glossary/u""",
                """Climate Change: Climate change refers to a statistically significant variation in either the mean state of the climate or in its variability, persisting for an extended period (typically decades or longer). Climate change may be due to natural internal processes or external forcings, or to persistent anthropogenic changes in the composition of the atmosphere or in land use. The United Nations Framework Convention on Climate Change (UNFCCC), however, does make a distinction between "climate variability" attributable to natural causes and "climate change" attributable to human activities altering the atmospheric composition. In the UNFCCC's Article 1, "climate change" is defined as: "a change of climate which is attributed directly or indirectly to human activity that alters the composition of the global atmosphere and which is in addition to natural climate variability observed over comparable time periods." (IPCC, 2001) Climate Feedback: The influence of a climate-related process on another that in turn influences the original process. For example, a positive climate feedback is an increase in temperature leading to a decrease in ice cover, which in turn leads to a decrease of reflected radiation (resulting in an increase in temperature). An example of a negative climate feedback is an increase in the Earth's surface temperature, which may locally increase cloud cover, which may reduce the temperature of the surface. (IPCC, 2001) Climatic Hazards: include increasing frequency of extreme weather events (floods, hurricanes, tornados, droughts), increasing summer temperatures, lower level of precipitation during main growing seasons, changes in streamflow, changes in snowfall Climate System: The system consisting of the atmosphere (gases), hydrosphere (water), lithosphere (solid rocky part of the Earth), and biosphere (living) that determine the Earth's climate. (NOAA, 2005) Climate Variability: Climate variability refers to variations in the mean state and other statistics (such as standard deviations, the occurrence of extremes, etc.) of the climate on all temporal and spatial scales beyond that of individual weather events. Variability may be due to natural internal processes within the climate system (internal variability), or to variations in natural or anthropogenic external forces (external variability). (IPCC, 2001) Climate: Climate in a narrow sense is usually defined as the 'average weather', or more rigorously, as the statistical description in terms of the mean and variability of relevant quantities over a period of time ranging from months to thousands or millions of years. Climate in a wider sense is the state, including a statistical description, of the climate system. The classical period of time is 30 years, as defined by the World Meteorological Organization (WMO). (IPCC, 2001) Climatic Variable: Qualitative classification of a weather element (e.g. temperature, precipitation, wind, humidity, etc.) at a place over a period of time. (NOAA, 2005) Coping Capacity: The means by which people or organizations use available resources and abilities to face adverse consequences that could lead to disaster. In general, this involves managing resources, both in normal times as well as during crises or adverse conditions. The strengthening of coping capacities usually builds resilience to withstand the effects of natural and human-induced hazards (UN/ISDR 2004). Capacity refers to the manner in which people and organizations use existing resources to achieve various beneficial ends during unusual, abnormal, and adverse conditions of a disaster event or process. The strengthening of coping capacities usually builds resilience to withstand the effects of natural and other hazards. (European Spatial Planning Observation Network) Critical Threshold: The point at which an activity faces an unacceptable level of harm, such as a change from profit to loss on a farm due to decreased water availability, or coastal flooding exceeding present planning limits. It occurs when a threshold q.v. is reached at which ecological or socioeconomic change is damaging and requires a policy response. (UNDP, 2005) Development Pathway: An evolution based on an array of technological, economic, social, institutional, cultural and biophysical characteristics that determine the interactions between human and natural systems, including production and consumption patterns in all countries, over time at a particular scale. (IPCC, 2007) Extreme Event: An extreme weather event refers to meteorological conditions that are rare for a particular place and/or time, such as an intense storm or heat wave. An extreme climate event is an unusual average over time of a number of weather events, for example heavy rainfall over a season.""",
                """ "Climate" means the prevailing weather conditions in a specific place over a long period of time. "Climate change" refers to the long-term shift in weather patterns such as temperature and precipitation. Increasing temperatures and other extreme weather events are being observed in Canada and around the world. In 2023, the hottest year on record globally, the Intergovernmental Panel on Climate Change (IPPC) declared that human activities are unequivocally causing climate change as a result of greenhouse gas emissions. Toronto is experiencing weather that is hotter, wetter, and wilder, and these conditions are expected to worsen. The number of days per year with temperatures above 30°C has already increased from an average of 8 days in the 1950s to about 18 days per year currently. Data suggests that if global emissions remain on their current path this could increase to 29 days by the 2030s (2021-2050), and 54 days by the 2060s (20512080) 1 . As well, data suggests that by 2080 Toronto will experience an increase in annual precipitation of 19%, and extreme rainstorms with 30% more rainfall than the historical baseline (1971-2000), which are expected to lead to flooding and associated infrastructure damage, injuries, habitat degradation, degraded water quality, soil erosion and disruptions to services and the economy.""",
            ]
    response1 = """Climate change refers to long-term shifts in temperatures and weather patterns. Here’s a simple breakdown to help you understand:
What is Weather vs. Climate?
Weather: This refers to short-term atmospheric conditions in a specific place at a specific time. For example, it’s raining today in Toronto.
Climate: This is the average weather conditions in a place over a long period, typically 30 years or more. It includes the average temperature, rainfall, and other weather patterns over decades.
What is Climate Change?
Climate change is a significant change in the state of the climate that can be identified by changes in the mean state and/or its variability. This can be due to:

Natural causes: Things like volcanic eruptions or natural cycles in the Earth's orbit around the sun.
Human activities: Mainly burning fossil fuels (like coal, oil, and gas) for energy, deforestation, and other activities that increase the levels of greenhouse gases in the atmosphere.
Key Points About Climate Change:
Greenhouse Gases: These are gases like carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O) that trap heat in the Earth's atmosphere, leading to warming. Human activities have increased the levels of these gases.
Global Warming: This is a specific aspect of climate change, referring to the overall increase in Earth’s average surface temperature due to human activities.
Impacts: Climate change can lead to more frequent and severe weather events, like heatwaves, floods, and droughts. It can also affect ecosystems, agriculture, and human health.
Why Should We Care?
Climate change poses risks to:

Health: Extreme weather can lead to injuries, illnesses, and even deaths.
Economy: Damage to infrastructure, agriculture, and fisheries can have significant economic impacts.
Environment: It can lead to loss of biodiversity, as plants and animals struggle to adapt to changing conditions.
What Can We Do?
Everyone can take action to help combat climate change:

Reduce, Reuse, Recycle: Less waste means less need for production and less pollution.
Use Less Fossil Fuels: Opt for public transport, carpool, bike, or walk whenever possible. Consider renewable energy sources like solar or wind power.
Plant Trees: Trees absorb CO2, a major greenhouse gas.
Educate Others: Sharing what you learn can help others understand the importance of addressing climate change.
By understanding climate change and taking action, we can help protect our planet for future generations.

Actionable Steps:
Limit Energy Use: Turn off lights when not in use, use energy-efficient appliances.
Support Renewable Energy: Advocate for and support policies that promote renewable energy.
Reduce Meat Consumption: Meat production is a significant source of greenhouse gases; reducing it can lower your carbon footprint.
Participate in Local Initiatives: Join or support local efforts to combat climate change, like tree-planting drives or community clean-ups.
Remember, every small action counts! Together, we can make a big difference."""

    response2 = """Climate change is a long-term shift in weather patterns, such as temperature and precipitation. It's a complex issue that can be caused by both natural processes and human activities.
What is climate?
Climate is the typical weather that a location experiences over a long period of time. For example, if you live in a place where it's usually hot and sunny, that's your climate.

What is weather?
Weather is the conditions of the atmosphere at a particular time and place. For example, if it's raining in your town today, that's the weather.

What is climate change?
Climate change is when these typical weather conditions start to change. This can be caused by natural processes, like volcanic eruptions, or human activities, like burning fossil fuels.

What are the effects of climate change?
Climate change can have lots of different effects, including:

Rising temperatures
Changes in rainfall
More extreme weather events, like floods, hurricanes, and droughts
Species loss and extinction
Damage to infrastructure
Disruptions to services and the economy
What can we do about climate change?
The Paris Agreement, adopted in 2015, is a legally binding international treaty on climate change. It aims to limit global warming to well below 2°C, and preferably to a 1.5°C increase, above pre-industrial levels.

There are lots of things we can do to help tackle climate change, like:

Using renewable energy sources, like solar and wind power
Reducing our use of fossil fuels
Protecting and restoring natural habitats
Educating ourselves and others about the issue"""

    try:
        logger.info('--checking hallucination--')

        load_environment()
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("Cohere API key not found in environment")
        
        await check_hallucination(question, response1, context, cohere_api_key)
        
        #await check_hallucination(question, response2, context, cohere_api_key)

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(test_hallucination_guard())
    print('processing time: ',time.time()-start_time)
    