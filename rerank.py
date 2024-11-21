import os
import cohere
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_docs_for_rerank(docs_to_rerank: List[Dict]) -> List[Dict]:
    """Prepare documents for reranking."""
    prepared_docs = []
    
    for doc in docs_to_rerank:
        try:
            # Get content from either content or chunk_text field
            content = doc.get('content', doc.get('chunk_text', ''))
            if not content.strip():
                logger.warning("Empty content found, skipping document")
                continue
                
            # Clean the content
            content = content.replace('\\n', ' ').replace('\\"', '"').strip()
            
            # Create the document for reranking
            prepared_doc = {
                'text': content,
                'title': doc.get('title', 'No Title'),
                'url': doc.get('url', [''])[0] if isinstance(doc.get('url', []), list) else doc.get('url', '')
            }
            
            # Store original document structure
            prepared_doc['original'] = doc
            
            prepared_docs.append(prepared_doc)
            
        except Exception as e:
            logger.error(f"Error preparing document for rerank: {str(e)}")
            continue
            
    return prepared_docs

def rerank_fcn(query: str, docs_to_rerank: List[Dict], top_k: int) -> List[Dict]:
    """Returns reranked documents that were retrieved."""
    try:
        logger.debug(f"Reranking {len(docs_to_rerank)} documents")
        logger.debug(f"Sample document structure: {docs_to_rerank[0] if docs_to_rerank else 'No docs'}")
        
        # Prepare documents for reranking
        prepared_docs = prepare_docs_for_rerank(docs_to_rerank)
        
        if not prepared_docs:
            logger.error("No valid documents to rerank")
            return []
            
        # Initialize Cohere client
        load_dotenv(".env")
        COHERE_API_KEY = os.getenv('COHERE_API_KEY')
        cohere_client = cohere.Client(COHERE_API_KEY)
        
        # Log the documents being sent to rerank
        logger.debug(f"Sending {len(prepared_docs)} documents to rerank")
        logger.debug(f"First document text: {prepared_docs[0]['text'][:100]}...")
        
        # Perform reranking
        rerank_results = cohere_client.rerank(
            query=query,
            documents=[doc['text'] for doc in prepared_docs],  # Send only the text content
            top_n=top_k,
            model="rerank-multilingual-v3.0"
        )
        
        # Reconstruct documents with original structure
        reranked_docs = []
        for result in rerank_results.results:
            original_doc = prepared_docs[result.index]['original']
            # Update score from reranking
            original_doc['score'] = result.relevance_score
            reranked_docs.append(original_doc)
            
        logger.info(f"Successfully reranked {len(reranked_docs)} documents")
        
        return reranked_docs
        
    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Test code
    test_docs = [
        {
            'title': 'Test Document 1',
            'content': 'This is some test content about climate change.',
            'url': ['http://example.com/1']
        },
        {
            'title': 'Test Document 2',
            'content': 'More test content about global warming.',
            'url': ['http://example.com/2']
        }
    ]
    
    try:
        result = rerank_fcn("climate change effects", test_docs, 2)
        print("Reranking successful!")
        print(f"Number of reranked documents: {len(result)}")
        if result:
            print(f"First document score: {result[0]['score']}")
    except Exception as e:
        print(f"Test failed: {str(e)}")