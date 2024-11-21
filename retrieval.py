import os
import time
import logging
import warnings
import numpy as np
from typing import List, Dict, Any
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out the specific tokenizer warning
warnings.filterwarnings("ignore", message="You're using a XLMRobertaTokenizerFast tokenizer")

def get_query_embeddings(query: str, embed_model) -> tuple:
    """Get dense and sparse embeddings for a query."""
    embeddings = embed_model.encode(
        [query], 
        return_dense=True, 
        return_sparse=True, 
        return_colbert_vecs=False
    )
    
    query_dense_embeddings = embeddings['dense_vecs']
    query_sparse_embeddings_lst = embeddings['lexical_weights']
    
    query_sparse_embeddings = []
    for sparse_embedding in query_sparse_embeddings_lst:
        sparse_dict = {}
        sparse_dict['indices'] = [int(index) for index in list(sparse_embedding.keys())]
        sparse_dict['values'] = [float(v) for v in list(sparse_embedding.values())]
        query_sparse_embeddings.append(sparse_dict)
    
    if isinstance(query_dense_embeddings, np.ndarray):
        query_dense_embeddings = query_dense_embeddings.astype(float)
    
    return query_dense_embeddings, query_sparse_embeddings

def weight_by_alpha(sparse_embedding: Dict, dense_embedding: List[float], alpha: float) -> tuple:
    """Weight the sparse and dense embeddings."""
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    hsparse = {
        'indices': sparse_embedding['indices'],
        'values': [float(v * (1 - alpha)) for v in sparse_embedding['values']]
    }
    hdense = [float(v * alpha) for v in dense_embedding]
    return hsparse, hdense

def issue_hybrid_query(index, sparse_embedding: Dict, dense_embedding: List[float], 
                      alpha: float, top_k: int):
    """Execute hybrid search query on Pinecone index."""
    scaled_sparse, scaled_dense = weight_by_alpha(sparse_embedding, dense_embedding, alpha)
    
    result = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True
    )
    return result

def get_hybrid_results(index, query: str, embed_model, alpha: float, top_k: int):
    """Get hybrid search results."""
    query_dense_embeddings, query_sparse_embeddings = get_query_embeddings(query, embed_model)
    return issue_hybrid_query(
        index, 
        query_sparse_embeddings[0], 
        query_dense_embeddings[0], 
        alpha, 
        top_k
    )
def clean_markdown_content(content: str) -> str:
    """Clean markdown formatting from content."""
    import re
    
    # Remove markdown table header and separator rows
    content = re.sub(r'\|[- |]+\|', '', content)
    
    # Extract meaningful text from table rows
    table_texts = []
    for line in content.split('\n'):
        if line.strip().startswith('|'):
            # Extract text between pipes, remove leading/trailing whitespace
            cells = [cell.strip() for cell in line.split('|')]
            # Filter out empty cells and join meaningful text
            meaningful_text = ' '.join(cell for cell in cells if cell)
            if meaningful_text:
                table_texts.append(meaningful_text)
    
    # If we found table text, join it together
    if table_texts:
        content = ' '.join(table_texts)
    
    # Remove multiple newlines and spaces
    content = re.sub(r'\n+', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    
    # Clean up specific characters
    content = (content.replace('\\n', ' ')
              .replace('\\"', '"')
              .replace('\\\'', "'")
              .replace('\\_{', '_')
              .replace('\\', '')  # Remove remaining backslashes
              .strip())
    
    return content

def process_search_results(search_results) -> List[Dict]:
    """
    Process search results into a standardized format with deduplication and content cleaning.
    """
    processed_docs = []
    seen_titles = set()  # For deduplication
    
    for match in search_results.matches:
        try:
            # Extract metadata
            title = match.metadata.get('title', 'No Title')
            
            # Skip if we've seen this title
            if title in seen_titles:
                continue
                
            # Get and clean content
            content = match.metadata.get('chunk_text', '')
            if not content:
                logger.warning(f"No content found for document: {title}")
                continue
                
            # Clean content
            content = clean_markdown_content(content)
            
            # Skip if content is too short after cleaning
            if len(content.strip()) < 10:
                logger.warning(f"Content too short after cleaning for document: {title}")
                continue
                
            # Create document
            doc = {
                'title': title,
                'content': content,
                'score': float(match.score),
                'section_title': match.metadata.get('section_title', '').strip(),
                'segment_id': match.metadata.get('segment_id', ''),
                'doc_keywords': match.metadata.get('doc_keywords', []),
                'segment_keywords': match.metadata.get('segment_keywords', []),
                'url': match.metadata.get('url', [])
            }
            
            processed_docs.append(doc)
            seen_titles.add(title)
            logger.debug(f"Successfully processed document: {title}")
                
        except Exception as e:
            logger.warning(f"Error processing match: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(processed_docs)} documents")
    return processed_docs

def format_document_output(doc: Dict) -> str:
    """Format document for display with better content preview."""
    output = [
        f"\nDocument:",
        f"Title: {doc['title']}",
        f"Score: {doc['score']:.3f}",
        f"Section: {doc['section_title']}",
    ]
    
    # Add keywords if available
    if doc['doc_keywords']:
        output.append(f"Keywords: {', '.join(doc['doc_keywords'][:5])}...")
        
    # Add clean content preview with better truncation
    content = doc['content']
    if len(content) > 300:
        # Try to break at a sentence or punctuation
        breakpoints = ['. ', '? ', '! ', '; ', ', ']
        truncated = False
        for point in breakpoints:
            last_point = content[:300].rfind(point)
            if last_point > 0:
                content = content[:last_point + 1]
                truncated = True
                break
        if not truncated:
            # If no good breakpoint found, break at word boundary
            content = content[:300].rsplit(' ', 1)[0]
        content += "..."
            
    output.append(f"\nContent preview: {content}")
    
    # Add source if available
    if doc['url'] and doc['url'][0]:
        output.append(f"Source: {doc['url'][0]}")
        
    return "\n".join(output)

def test_retrieval():
    """Test the document retrieval process"""
    try:
        # Initialize
        print("\n=== Testing Document Retrieval ===")
        from dotenv import load_dotenv
        load_dotenv(".env")
        
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        # Setup
        print("\nInitializing components...")
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        index = pinecone_client.Index("climate-change-adaptation-index-10-24-prod")
        embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        
        # Test query
        query = "What is climate change?"
        print(f"\nProcessing query: {query}")
        
        # Get results
        start_time = time.time()
        hybrid_results = get_hybrid_results(
            index=index,
            query=query,
            embed_model=embed_model,
            alpha=0.4,
            top_k=5
        )
        search_time = time.time() - start_time
        
        # Process results
        processed_docs = process_search_results(hybrid_results)
        
        # Display results
        print(f"\nRetrieved and processed {len(processed_docs)} documents in {search_time:.2f} seconds:")
        
        for doc in processed_docs:
            print(format_document_output(doc))
            print("-" * 80)
            
        return processed_docs
        
    except Exception as e:
        logger.error(f"Error in test_retrieval: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    finally:
        print("\nScript execution completed")