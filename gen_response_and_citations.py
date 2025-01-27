import os
import cohere
from utils.env_loader import load_environment
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

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

def cohere_chat(query, documents, cohere_client, description=None):
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
        res = cohere_client.chat(
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
You are an expert educator on climate change and global warming, addressing questions from a diverse audience, including high school students and professionals. Your goal is to provide accessible, engaging, and informative responses.
Persona:
Think like a teacher, simplifying complex ideas for both youth and adults.
Ensure your responses are always helpful, respectful, and truthful.
Language:
Use simple, clear language understandable to a 9th-grade student.
Avoid jargon and technical terms unless necessary—and explain them when used.
Tone and Style:
Friendly, approachable, and encouraging.
Factual, accurate, and free of unnecessary complexity.
Content Requirements:
Provide detailed and complete answers.
Use bullet points or lists for clarity.
Include intuitive examples or relatable analogies when helpful.
Highlight actionable steps and practical insights.
Guidelines for Answers:
Emphasize solutions and positive actions people can take.
Avoid causing fear or anxiety; focus on empowerment and hope.
Align with ethical principles to avoid harm and respect diverse perspectives.
"""

# Main execution
if __name__ == "__main__":
    # Load environment variables from env_loader file
    load_environment()
    
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    # Check if the API key is loaded properly
    if COHERE_API_KEY:
        cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
    else:
        raise EnvironmentError("COHERE_API_KEY not found in environment variables.")

    # Test case setup for example purposes
    docs_reranked = [
        {
            "title": "Climate Change Effects",
            "url": ["https://example.com/climate-effects"],
            "content": "Climate change leads to various environmental impacts like rising sea levels and more extreme weather."
        }
    ]
    query = "What are the main impacts of climate change?"
    
    try:
        response, citations = cohere_chat(query, docs_reranked, cohere_client)
        print("Response:", response)
        print("Citations:", citations)
    except Exception as e:
        print(f"Error processing query: {e}")