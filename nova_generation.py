from utils.env_loader import load_environment
import time
import json
import textwrap
from nova_flow import BedrockModel
import logging

logger = logging.getLogger(__name__)

# Persona prompt for climate change education
default_persona_prompt = """
    You are an expert educator on climate change and global warming, addressing questions from a diverse audience, 
    including high school students and professionals. Your goal is to provide accessible, engaging, and informative responses.

    Persona:
    - Think like a teacher, simplifying complex ideas for both youth and adults.
    - Ensure your responses are always helpful, respectful, and truthful.

    Language:
    - Use simple, clear language understandable to a 9th-grade student.
    - Avoid jargon and technical terms unless necessary—and explain them when used.

    Tone and Style:
    - Friendly, approachable, and encouraging.
    - Factual, accurate, and free of unnecessary complexity.

    Content Requirements:
    - Provide detailed and complete answers.
    - Use bullet points or lists for clarity.
    - Include intuitive examples or relatable analogies when helpful.
    - Highlight actionable steps and practical insights.

    Guidelines for Answers:
    - Emphasize solutions and positive actions people can take.
    - Avoid causing fear or anxiety; focus on empowerment and hope.
    - Align with ethical principles to avoid harm and respect diverse perspectives.
    """

def get_citation_info(docs):
    """Prepare documents for Cohere chat."""
    citations = []
    logger.debug(f"Processing {len(docs)} documents")
    
    for doc in docs:
        try:
            # Extract required fields
            title = doc.get('title', '')
            content = doc.get('content', '')  # Primary content field
            if not content:
                content = doc.get('chunk_text', '')  # Fallback content field
                
            # Get URL(s)
            url = doc.get('url', [])
            if isinstance(url, list) and url:
                url = url[0]
            elif isinstance(url, str):
                url = url
            else:
                url = ''
                
            # Validation
            if not title or not content:
                logger.warning(f"Missing required fields - Title: {bool(title)}, Content: {bool(content)}")
                continue
                
            # Clean content
            content = content.replace('\\n', ' ').replace('\\"', '"').strip()
            if len(content) < 10:
                logger.warning(f"Content too short for document: {title}")
                continue
                
            citation = {
                    "title": f"{title}: {url}" if url else title,
                    "snippet": content
            }
            
            logger.debug(f"Processed document - Title: {title}")
            logger.debug(f"Content length: {len(content)}")
            
            citations.append(citation)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            continue
    
    if citations:
        logger.info(f"Successfully processed {len(citations)} documents")
    else:
        logger.error("No documents were successfully processed")
        
    return citations

class NovaChat(BedrockModel):
    def __init__(self, model_id='amazon.nova-micro-v1:0', region_name='us-east-1'):
        """
        Initializes the NovaChat class, inheriting from BedrockModel.
        
        Args:
            model_id (str): The ID of the model to interact with.
            region_name (str): AWS region where Bedrock is deployed.
        """
        super().__init__(model_id, region_name)  # Call parent class initialization

    def nova_chat(self, query, documents, persona_prompt=default_persona_prompt, description=None):
        """
        Sends a contextual query to the specified Amazon Bedrock model using the Converse API.

        Args:
            model_id (str): The ID of the Bedrock model to use (e.g., "amazon.nova-micro-v1:0").
            documents (list): A list of document dictionaries with 'source' and 'content' keys.
            query (str): The specific question or task to be addressed.
            persona_prompt (str): A detailed prompt defining the model's persona and response guidelines.

        Returns:
            dict: The model's response, including the generated output and other details.
        """

        citations = get_citation_info(documents)

        # Construct the messages payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"{persona_prompt}" + 
                                (f" [description: {description}]" if description else "") + 
                                "\n\nContext Documents:"
                    }
                ]
            }
        ]

        # Add context documents
        for citation in citations:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "text": f"Document content:\n{textwrap.fill(citation['snippet'], width=100)}"
                    }
                ]
            })

        # Add the final query
        messages.append({
            "role": "user",
            "content": [
                {
                    "text": query
                }
            ]
        })

        # Construct the payload
        payload = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 2000,  # Increased to accommodate more detailed responses
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        try:
            # Send the POST request
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            response = response_body['output']['message']['content'][0]['text']
            return response, citations
        except Exception as e:
            raise RuntimeError(f"An error occurred while invoking the model: {e}")

def main():
    start_time = time.time()

    # Load environment variables from env_loader file
    load_environment()
    
    # Example documents (you would replace these with actual document content)
    documents = [
        {
            "title": "Health of Canadians in Changing Climate Report",
            "url": "/content/drive/MyDrive/PDFs_10_24/step 1_try2/Health of Canadians in Changing Climate Report.md",
            "content": "Climate change - A persistent, long-term change in the state of the climate, measured by changes in the mean state and/or its variability. Climate change may be due to natural internal processes, natural external forcings such as volcanic eruptions and modulations of the solar cycle, or to persistent anthropogenic changes in the composition of the atmosphere or in land use (IPCC, 2014)."
        },
        {
            "title": "CANADIAN COMMUNITIES’ GUIDEBOOK FOR ADAPTATION TO CLIMATE CHANGE Including an approach to generate mitigation co-benefits in the context of sustainable development REPORT",
            "url": "https://publications.gc.ca/site/eng/9.834255/publication.html",
            "content": """Climate Change: Climate change refers to a statistically significant variation in either the mean state of the climate or in its variability, persisting for an extended period (typically decades or longer). Climate change may be due to natural internal processes or external forcings, or to persistent anthropogenic changes in the composition of the atmosphere or in land use. The United Nations Framework Convention on Climate Change (UNFCCC), however, does make a distinction between "climate variability" attributable to natural causes and "climate change" attributable to human activities altering the atmospheric composition. In the UNFCCC\'s Article 1, "climate change" is defined as: "a change of climate which is attributed directly or indirectly to human activity that alters the composition of the global atmosphere and which is in addition to natural climate variability observed over comparable time periods." (IPCC, 2001) Climate Feedback: The influence of a climate-related process on another that in turn influences the original process. For example, a positive climate feedback is an increase in temperature leading to a decrease in ice cover, which in turn leads to a decrease of reflected radiation (resulting in an increase in temperature). An example of a negative climate feedback is an increase in the Earth\'s surface temperature, which may locally increase cloud cover, which may reduce the temperature of the surface. (IPCC, 2001) Climatic Hazards: include increasing frequency of extreme weather events (floods, hurricanes, tornados, droughts), increasing summer temperatures, lower level of precipitation during main growing seasons, changes in streamflow, changes in snowfall Climate System: The system consisting of the atmosphere (gases), hydrosphere (water), lithosphere (solid rocky part of the Earth), and biosphere (living) that determine the Earth\'s climate. (NOAA, 2005) Climate Variability: Climate variability refers to variations in the mean state and other statistics (such as standard deviations, the occurrence of extremes, etc.) of the climate on all temporal and spatial scales beyond that of individual weather events. Variability may be due to natural internal processes within the climate system (internal variability), or to variations in natural or anthropogenic external forces (external variability). (IPCC, 2001) Climate: Climate in a narrow sense is usually defined as the \'average weather\', or more rigorously, as the statistical description in terms of the mean and variability of relevant quantities over a period of time ranging from months to thousands or millions of years. Climate in a wider sense is the state, including a statistical description, of the climate system. The classical period of time is 30 years, as defined by the World Meteorological Organization (WMO). (IPCC, 2001) Climatic Variable: Qualitative classification of a weather element (e.g. temperature, precipitation, wind, humidity, etc.) at a place over a period of time. (NOAA, 2005) Coping Capacity: The means by which people or organizations use available resources and abilities to face adverse consequences that could lead to disaster. In general, this involves managing resources, both in normal times as well as during crises or adverse conditions. The strengthening of coping capacities usually builds resilience to withstand the effects of natural and human-induced hazards (UN/ISDR 2004). Capacity refers to the manner in which people and organizations use existing resources to achieve various beneficial ends during unusual, abnormal, and adverse conditions of a disaster event or process. The strengthening of coping capacities usually builds resilience to withstand the effects of natural and other hazards. (European Spatial Planning Observation Network) Critical Threshold: The point at which an activity faces an unacceptable level of harm, such as a change from profit to loss on a farm due to decreased water availability, or coastal flooding exceeding present planning limits. It occurs when a threshold q.v. is reached at which ecological or socioeconomic change is damaging and requires a policy response. (UNDP, 2005) Development Pathway: An evolution based on an array of technological, economic, social, institutional, cultural and biophysical characteristics that determine the interactions between human and natural systems, including production and consumption patterns in all countries, over time at a particular scale. (IPCC, 2007) Extreme Event: An extreme weather event refers to meteorological conditions that are rare for a particular place and/or time, such as an intense storm or heat wave. An extreme climate event is an unusual average over time of a number of weather events, for example heavy rainfall over a season."""
        },
        {
            "title":"Planning to be a Part of my Region! Grade 9 lesson plans designed for students in the Region of Peel",
            "url": "https://www.peelregion.ca/planning/teaching-planning/pdfs/Grade9forWeb.pdf",
            "content":"""Climate change: Climate change refers to a change in the state of the climate that can be identified (e.g. by using statistical tests) by changes in the mean and/or the variability of its properties, and that persists for an extended period, typically decades or longer. Climate change may be due to natural internal processes or external forcing factors, or to persistent anthropogenic changes in the composition of the atmosphere or in land use. Note that the United Nations Framework Convention on Climate Change (UNFCCC) defines climate change as "a change of climate which is attributed directly or indirectly to human activity that alters the composition of the global atmosphere and which is in addition to natural climate variability observed over comparable time periods. " The UNFCCC thus makes a distinction between climate change attributable to human activities altering the atmospheric composition, and climate variability attributable to natural causes. (From Impacts to Adaptation : Canada in a Changing Climate, 2007, http://adaptation.nrcan.gc.ca/assess/2007/ch11/index_e.php#R) Greenhouse effect: Greenhouse gases effectively absorb infrared radiation, emitted by the Earth\'s surface, by the atmosphere itself due to the same gases and by clouds. Atmospheric radiation is emitted to all sides, including downward to the Earth\'s surface. Thus, greenhouse gases trap heat within the surface-troposphere system. This is called the greenhouse effect. http://www.ipcc.ch/publications_and_data/ar4/wg3/en/annex1sglossary-e-i.html Greenhouse gas (GHG ): Gaseous constituents of the atmosphere, both natural and anthropogenic, that absorb and emit radiation at specific wavelengths within the spectrum of infrared radiation emitted by the Earth \'s surface, by the atmosphere itself and by clouds. Water vapour (H2O), carbon dioxide (CO2), nitrous oxide (N2O), methane (CH4) and ozone (O3) are the primary greenhouse gases in the Earth\'s atmosphere. In addition, there are a number of entirely human-made greenhouse gases in the atmosphere, such as the halocarbons and other chlorine- and bromine-containing substances. (From Impacts to Adaptation : Canada in a Changing Climate, 2007, http://adaptation.nrcan.gc.ca/assess/2007/ch11/index_e.php#R) Mitigation: Initiatives and measures to reduce the vulnerability of natural and human systems against actual or expected climate change effects. (IPCC, 2008) Natural Systems: Includes the natural environment (forests, wetlands, wildlife, etc.) plus human activities to monitor, protect and enhance the natural environment and educate the public on the current and future state of our natural systems. http://www.peelregion.ca/planning/climatechange/reports/pdf/climate-chan-strat-bgr.pdf Public Health : Involves the combination of programs, services and policies that protect and promote the health of all. Public health is involved in the enhancement of the health status of the population; reduction of disparities in health status among individual/groups within that population; preparation for and response to outbreaks and emergencies; and enhancing the sustainability of the healthcare system. http://www.peelregion.ca/planning/climatechange/reports/pdf/climate-chan-strat-bgr.pdf Urban forest: The trees, forests and associated organisms that grow near buildings and in gardens, green spaces, parks and golf courses located in village, town, suburban and urban areas. http://canadaforests.nrcan.gc.ca/glossary/u"""
        },
        {
            "title":"REPORT FOR ACTION Toronto's Climate Change Readiness-Updates oncommitments and a refreshed mandate for coordinating resilience activities",
            "url": "https://www.toronto.ca/legdocs/mmis/2024/ie/bgrd/backgroundfile-244181.pdf",
            "content":""""Climate" means the prevailing weather conditions in a specific place over a long period of time. "Climate change" refers to the long-term shift in weather patterns such as temperature and precipitation. Increasing temperatures and other extreme weather events are being observed in Canada and around the world. In 2023, the hottest year on record globally, the Intergovernmental Panel on Climate Change (IPPC) declared that human activities are unequivocally causing climate change as a result of greenhouse gas emissions. Toronto is experiencing weather that is hotter, wetter, and wilder, and these conditions are expected to worsen. The number of days per year with temperatures above 30°C has already increased from an average of 8 days in the 1950s to about 18 days per year currently. Data suggests that if global emissions remain on their current path this could increase to 29 days by the 2030s (2021-2050), and 54 days by the 2060s (20512080) 1 . As well, data suggests that by 2080 Toronto will experience an increase in annual precipitation of 19%, and extreme rainstorms with 30% more rainfall than the historical baseline (1971-2000), which are expected to lead to flooding and associated infrastructure damage, injuries, habitat degradation, degraded water quality, soil erosion and disruptions to services and the economy."""
        },
        {
            "title":"11356_2022_Article_19718",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8978769/",
            "content":"""Worldwide observed and anticipated climatic changes for the twenty-first century and global warming are significant global changes that have been encountered during the past 65 years. Climate change (CC) is an inter-governmental complex challenge globally with its influence over various components of the ecological, environmental, socio-political, and socio-economic disciplines (Adger et al. 2005; Leal Filho et al. 2021; Feliciano et al. 2022). Climate change involves heightened temperatures across numerous worlds (Battisti and Naylor 2009; Schuurmans 2021; Weisheimer and Palmer 2005; Yadav et al. 2015). With the onset of the industrial revolution, the problem of earth climate was amplified manifold (Leppänen et al. 2014). It is reported that the immediate attention and due steps might increase the probability of overcoming its devastating impacts. It is not plausible to interpret the exact consequences of climate change (CC) on a sectoral basis (Izaguirre et al. 2021; Jurgilevich et al. 2017), which is evident by the emerging level of recognition plus the inclusion of climatic uncertainties at both local and national level of policymaking (Ayers et al. 2014). Climate change is characterized based on the comprehensive long-haul temperature and precipitation trends and other components such as pressure and humidity level in the surrounding environment. Besides, the irregular weather patterns, retreating of global ice sheets, and the corresponding elevated sea level rise are among the most renowned international and domestic effects of climate change (Lipczynska-Kochany 2018; Michel et al. 2021; Murshed and Dao 2020). Before the industrial revolution, natural sources, including volcanoes, forest fires, and seismic activities, were regarded as the distinct sources of greenhouse gases (GHGs) such as CO$_2}$, CH$_4}$, N$_2}$O, and H$_2}$O into the atmosphere (Murshed et al. 2020; Hussain et al. 2020; Sovacool et al. 2021; Usman and Balsalobre-Lorente 2022; Murshed 2022). United Nations Framework Convention on Climate Change (UNFCCC) struck a major agreement to tackle climate change and accelerate and intensify the actions and investments required for a sustainable low-carbon future at Conference of the Parties (COP-21) in Paris on December 12, 2015. The Paris Agreement expands on the Convention by bringing all nations together for the first time in a single cause to undertake ambitious measures to prevent climate change and adapt to its impacts, with increased funding to assist developing countries in doing so. As so, it marks a turning point in the global climate fight. The core goal of the Paris Agreement is to improve the global response to the threat of climate change by keeping the global temperature rise this century well below 2 °C over pre-industrial levels and to pursue efforts to limit the temperature increase to 1.5° C (Sharma et al. 2020; Sharif et al. 2020; Chien et al. 2021."""
        },
        {
            "title":"12992_2021_Article_722",
            "url": "https://www.terrascope.com/blog/understanding-greenhouse-gases-and-climate-change",
            "content":"""Author Definition Todorov, A.V. (1986) [ 60 ] The concept of climate change is both complex and controversial. There is no unanimous opinion and agreement among climatologists on the definition of the term climate, not to mention climate change, the trend or climatic fluctuation. United Nations (Bodansky, 1993) [ 9 ] A variation in the climate attributed directly or indirectly to human activity that alters the composition of the world ' s atmosphere and that adds to the natural variability of the climate observed in comparable periods of time. Lorenz, E. (1995) [Lorenz EN: Climate is what you expect. Unpublished] Climate is the current distribution of a climate system over time that extends indefinitely into the future, so there is no talk of the existence of climate change. IPCC (Parry et al., 2007) [ 46 ] A change in the state of climate that can be identified (for example, by statistical tests) by changes in the average and / or the variability of its properties, and that persists for a prolonged period, usually of decades or more. Werndl, C. (2014) Different climatic distributions in two successive periods of time. Article 321 263 Book and Book Chapter 65 102 Review 14 30 Proceeding Paper 39 13"""
        },
        {
            "title":"TransformTO-Net-Zero-Framework-Technical-Report-Parts-1-2",
            "url": "https://www.toronto.ca/wp-content/uploads/2022/04/8f02-TransformTO-Net-Zero-Framework-Technical-Report-Parts-1-2.pdf",
            "content":"""Climate change is the greatest long-term global challenge that human society is facing. It is particularly complex because it occurs over a long time scale, has variable impacts globally and spatially, and requires rapid and radical changes to our energy, society, and economic systems. Human-induced climate change poses risks to health, economic growth, public safety, infrastructure, livelihoods, and the world's biodiversity and ecosystems. As local and global greenhouse gas (GHG) emissions increase, the Earth continues to warm at an unprecedented rate. In December 2015, the Paris Agreement was adopted at the COP21 by 197 countries. This legally binding international treaty on climate change set a goal to limit global warming to well below a 2°C, and preferably to a 1.5°C increase, above pre-industrial levels. 16 However, current global GHG emissions are not on a trajectory to meet these goals (Figure 8). Figure 8. Likelihood of limiting warming to 1.5°C given a global target of achieving net zero by 2040. 17 <!-- image --> Despite a temporary decline in global emissions in 2020 due to the COVID-19 pandemic, the world is heading for 3°C or more of warming. 18 This degree of warming threatens human health, economic well-being, and the survival of the natural systems that humans and eight million other plant and animal species-already increasingly at risk-depend upon. 19 Given the short timeline to achieve the required transformative changes to our economic, transportation, infrastructure, financial, and energy systems, COVID-19 recovery plans provide a unique opportunity to invest in and accelerate toward limiting global warming to 1.5°C"""
        },
        {
            "title": "Climate Change 2021-Summary for All",
            "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/outreach/IPCC_AR6_WGI_SummaryForAll.pdf",
            "content":"""No matter where we live, we all experience weather: how the conditions of our atmosphere change over minutes, hours, days, weeks. We also all experience climate: the weather of a place averaged over several decades. Climate change is when these averaged conditions start to change and its causes can be either natural or caused by human activities. Rising temperatures, variations in rainfall, increased extreme weather events are all examples of climate changes, but there are many others. Back in 1990, the first report by the Intergovernmental Panel on Climate Change (IPCC) concluded that human-caused climate change would soon become apparent but could not yet confirm that it was already happening. Now, some 30 years later, the evidence is overwhelming that human activities have changed the climate. Hundreds of scientists from all over the world come together to produce IPCC reports. They base their conclusions on several kinds of scientific evidence, including: - · Measurements or observations, sometimes spanning more than a century back in time; - · Paleo (very old) climate evidence from thousands or millions of years ago (for example: tree rings, rock or ice cores); - · Computer models that look at past, current and future changes (see box What are climate models?on page 9 ); - · Understanding of how the climate works (physical, chemical and biological processes). Since the IPCC first started, we now have much more data and better climate models. We understand more now about how the atmosphere interacts with the ocean, ice, snow, ecosystems and land of the Earth. Computer climate simulations have considerably improved and now provide past change and future projections that are much more detailed. Plus, we have now had decades more greenhouse gas emissions, making the effects of climate change more apparent (see box What are greenhouse gases?on page 6 ). As a result, the latest IPCC report is able to confirm and strengthen the conclusions from previous reports."""
        },
        {
            "title": "Project of Learning for a sustainable future-Chp1-What is Climate Change-Why Care",
            "url": "https://climatelearning.ca/wp-content/uploads/2023/10/3-6-Chapter-1-EN.pdf",
            "content":"""The difference between weather and climate is that whereas weather describes an event occurring at a particular time and place - a storm moving in over a city for example - climate describes the typical weather that a location experiences based on the study of weather conditions over long periods of time. An often heard expression is that \'climate is what you expect, and weather is what you get\'. ( Let\'s Talk Energy - Climate vs. Weather: A collaborative project with the Royal Canadian Geographical Society (RCGS) and Ingenium) To better understand the difference between climate and weather, watch this video by National Geographic that features Neil Degrasse Tyson. "A greenhouse is used to create a warmer growing environment for plants that would otherwise not survive in the colder conditions outdoors. In a greenhouse, energy from the sun passes through the glass as rays of light. This energy is absorbed by the plants, soil and other objects in the greenhouse. Much of this absorbed energy is converted to heat, which warms the greenhouse. The glass helps keep the greenhouse warm, by preventing the warmed air from escaping." (Ingenium, 2022) Some of the many impacts of climate change include: biodiversity, ecosystems, species loss and extinction. If the global community is able to limit the increase in temperature to 1.5 degrees, the impacts on terrestrial, freshwater and coastal ecosystems are expected to be lower. According to the Council of Canadian Academies\' expert panel on climate change risks and adaptation potential, Canada faces substantial risk with a likelihood of significant losses, damages, or disruptions in Canada over a 20 year timeframe in the following areas: agriculture and food; coastal communities; ecosystems; fisheries; forestry; geopolitical dynamics; governance and capacity; human health and wellness; Indigenous ways of life; northern communities; physical infrastructure; and water. Overall, Canadians are quite certain that climate change is happening. According to the national survey Canadians\' Perspectives on Climate Change & Education (2022) conducted by Learning for a Sustainable Future, 81% of all Canadians believe that climate change is happening. However, the population is less certain that humans are the primary cause of the warming climate; only 54% of respondents think that climate change is caused mostly by human activity. When this understanding is contrasted with the widespread scientific consensus that climate change is primarily caused by the human activity of burning fossil fuels, the urgent need for more comprehensive education on the subject is made clear. Another finding from the report, Canada, Climate Change and Education: Opportunities for Public and Formal Education , found that 46% of students ages 12-18 are categorized as "aware," meaning they understand that human-caused climate change is happening, but they do not believe that human efforts to stop it will be effective. This is an opportunity for schools to help students understand that there are strategies and solutions to address climate change if all sectors take action today"""
        }
    ]

    # Example query
    query = "what is climate change?"
    model = NovaChat()
    try:
        description = 'Please ensure strict factual accuracy'

        # Call the function to get the model's response
        response, citations = model.nova_chat(query, documents, description)
        
        # Extract and print the model's response
        #if 'output' in response and 'message' in response['output'] and 'content' in response['output']['message']:
        #   #print("Model Response:")
        #    for content_block in response['output']['message']['content']:
        #        if 'text' in content_block:
        #            print(content_block['text'])
        #else:
        #    print("No text response found in the model output.")
        print(response)
        print('processing time: ',time.time()-start_time)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    main()