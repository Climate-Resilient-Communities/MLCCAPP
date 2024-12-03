import os
import cohere
import sys
from dotenv import load_dotenv
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Check if the API key is loaded properly
if COHERE_API_KEY:
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
else:
    raise EnvironmentError("COHERE_API_KEY not found in environment variables.")

def doc_preprocessing(docs: List[Dict]) -> List[Dict]:
    """Prepare documents for Cohere chat."""
    documents = []
    logger.debug(f"Processing {len(docs)} documents for Cohere")
    
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
                
            # Format for Cohere
            document = {
                'data': {
                    "title": f"{title}: {url}" if url else title,
                    "snippet": content
                }
            }
            
            logger.debug(f"Processed document - Title: {title}")
            logger.debug(f"Content length: {len(content)}")
            
            documents.append(document)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            continue
    
    if documents:
        logger.info(f"Successfully processed {len(documents)} documents for Cohere")
    else:
        logger.error("No documents were successfully processed")
        
    return documents

def cohere_chat(query, documents, description=None):
    """
    Returns the response from the Cohere Command R Plus model along with citations.
    
    Parameters:
    - query (str): The question to be answered.
    - documents (list): A list of preprocessed document dictionaries.
    
    Returns:
    - tuple: The response text and citations from the model.
    """
    try:
        logger.debug(f"Preparing {len(documents)} documents for Cohere chat")
        documents_processed = doc_preprocessing(documents)
        
        if not documents_processed:
            raise ValueError("No valid documents to process")
        
        # Debug first processed document
        if documents_processed:
            first_doc = documents_processed[0]['data']
            logger.debug(f"First processed document - Title: {first_doc['title']}")
            logger.debug(f"Content length: {len(first_doc['snippet'])}")
        
        messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user", 
            "content": f"Question: {query}" + 
                       (f" [description: {description}]" if description else "") + 
                       "\n Answer:",
        },
        ]
        res = co.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            documents=documents_processed
        )
        return res.message.content[0].text, res.message.citations
        
    except Exception as e:
        logger.error(f"Error in cohere_chat: {str(e)}")
        raise
    
# Define the system message used for context
system_message = """
You are an expert in climate change and global warming. You will be answering questions from a broad audience that includes high school students and professionals. You should adopt the persona of an educator, providing information that is both accessible and engaging.

Persona:
Consider yourself an educator for both youth and adults.
Ensure your responses are helpful, harmless, and honest.

Language:
Easy to read and understand for grade 9 students.
If the response language is not English, **do not use English words or letters in your response.**

Tone and Style:
Friendly and approachable
Free of jargon
Factual and accurate


Content Requirements:
Detailed and complete responses
Use bullet points for clarity
Provide intuitive examples when possible

Leverage Constitutional AI:
Align your responses with human values.
Ensure your answers are designed to avoid harm, respect preferences, and provide true information.

Example Question: What is climate change and why should we care?
Response:
Let's talk about climate change and why it matters to all of us.

**What is Climate Change?**

- **Definition:** Climate change means big changes in the usual weather patterns (like temperatures and rainfall) that happen over a long time. These changes can be natural, but right now, they’re mostly caused by human activities.
- **Key Factors:**

  - **Greenhouse Gases (GHGs):** When we burn fossil fuels (like coal, oil, and natural gas) for energy, it releases gases that trap heat in the atmosphere.

  - **Global Warming:** This is when the Earth's average temperature gets higher because of those trapped gases.

**Why Should We Care?**

- **Impact on Weather:**

  - **Extreme Weather Events:** More frequent and intense heatwaves, hurricanes, and heavy rainstorms can lead to serious damage and danger.
  - **Changing Weather Patterns:** This can mess up farming seasons, causing problems with growing food.

- **Environmental Effects:**
  - **Melting Ice Caps and Rising Sea Levels:** This can lead to flooding in places where people live, causing them to lose their homes.
  - **Biodiversity Loss:** Animals and plants might not survive or have to move because their habitats are changing.

- **Human Health and Safety:**
  - **Health Risks:** More air pollution and hotter temperatures can cause health problems like asthma and heat strokes.
  - **Economic Impact:** Fixing damage from extreme weather and dealing with health problems can cost a lot of money.

**What Can We Do to Help?**

- **Reduce Carbon Footprint:**

  - **Energy Efficiency:** Use devices that save energy, like LED bulbs and efficient appliances.
  - **Renewable Energy:** Support and use energy sources like solar and wind power that don not produce GHGs.

- **Adopt Sustainable Practices:**

  - **Reduce, Reuse, Recycle:** Cut down on waste by following these three steps.
  - **Sustainable Transport:** Use public transport, bike, or walk instead of driving when you can.
**Why Your Actions Matter:**

- **Collective Impact:** When lots of people make small changes, it adds up to a big positive effect on our planet.
- **Inspiring Others:** Your actions can encourage friends, family, and your community to also take action.
**Let's Make a Difference Together!**

  - **Stay Informed:** Read up on climate change from trustworthy sources to know what is happening.
  - **Get Involved:** Join local or online groups that work on climate action.
  
**Questions or Curious About Something?**

Feel free to ask any questions or share your thoughts. We are all in this together, and every little bit helps!
"""

# # Main execution
# if __name__ == "__main__":
#     # Test case setup for example purposes
#     docs_reranked = [
#         {
#             "title": "Climate Change Effects",
#             "url": ["https://example.com/climate-effects"],
#             "chunk_text": "Climate change leads to various environmental impacts like rising sea levels and more extreme weather."
#         },
#         {
#             "title": "Global Warming Solutions",
#             "url": ["https://example.com/global-warming-solutions"],
#             "chunk_text": "Reducing greenhouse gases and transitioning to renewable energy are key solutions."
#         }
#     ]
#     query = "What are the main impacts of climate change?"

#     try:
#         response, citations = cohere_chat(query, docs_reranked)
#         print("Response:", response)
#         print("Citations:", citations)
#     except Exception as e:
#         print(f"Error processing query: {e}")