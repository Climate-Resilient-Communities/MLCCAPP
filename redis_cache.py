from redis import Redis
import json
from datetime import timedelta
import logging
import os
from utils.env_loader import load_environment

load_environment()

logger = logging.getLogger(__name__)

class ClimateCache:
    def __init__(self):
        """
        Initialize Redis connection using environment variables.
    
        Environment variables used:
        - REDIS_HOST: The Azure Redis Cache hostname (example: myapp.redis.cache.windows.net)
        - REDIS_PORT: The port number (usually 6380 for SSL)
        - REDIS_PASSWORD: The access key for authentication
    
        Returns:
            Redis: Configured Redis client instance
        """
        try:
            self.redis_client = Redis(
                host=os.getenv('REDIS_HOST'),
                port=int(os.getenv('REDIS_PORT', 6380)),  # Default SSL port for Azure Redis
                password=os.getenv('REDIS_PASSWORD'),
                ssl=True,  # Required for Azure Redis
                decode_responses=True  # Automatically decode responses to strings
            )

            # Verify connection
            self.redis_client.ping()
            print("Successfully connected to Redis cache")

        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.redis_client = None

    def get_citation_details(self, citation):
        """Safely extract citation details."""
        try:
            # Handle citations from the Cohere response
            if hasattr(citation, 'sources') and citation.sources:
                source = citation.sources[0]
                if hasattr(source, 'document'):
                    doc = source.document
                    return {
                        'title': doc.get('title', 'Untitled Source'),
                        'url': doc.get('url', ''),
                        'snippet': doc.get('snippet', '')
                    }
            # Handle dictionary-style citations
            elif isinstance(citation, dict):
                return {
                    'title': citation.get('title', 'Untitled Source'),
                    'url': citation.get('url', ''),
                    'snippet': citation.get('content', citation.get('snippet', ''))
                }
        except Exception as e:
            logger.error(f"Error processing citation: {str(e)}")
    
        return {
            'title': 'Untitled Source',
            'url': '',
            'snippet': ''
        }
    
    def save_to_cache(self, cache_key: str, result: dict, expiry_days: int = 7):
        """
        Save the query result to Redis cache.
    
        Args:
            cache_key (str): Unique key for the cache entry
            result (dict): The query result to cache
            expiry_days (int): Number of days to keep the cache entry
        """
        if self.redis_client is None:
            return False
        
        try:
            # Create a copy to avoid modifying the original result
            cache_result = result.copy()
        
            # Preprocess the 'citations' field to make it JSON-serializable
            if 'citations' in cache_result:
                cache_result['citations'] = [
                    self.get_citation_details(citation) for citation in cache_result['citations']
                ]
        
            # Serialize the result and save it to Redis with an expiration time
            self.redis_client.setex(
                cache_key,
                timedelta(days=expiry_days),
                json.dumps(cache_result)
            )
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def get_from_cache(self, cache_key: str) -> dict:
        """
        Retrieve the query result from Redis cache.
    
        Args:
            cache_key (str): Unique key for the cache entry
    
        Returns:
            dict: Cached result or None if not found
        """
        if self.redis_client is None:
            return None
    
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result['cache_hit'] = True
                return result
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
        
        return None
    
    