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

def cohere_chat(query, documents):
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
        
        res = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Question: {query}. \n Answer:",
                },
            ],
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