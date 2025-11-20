"""
OpenAI Embedding Generator for MONDO Diseases and Synonyms

Generates embeddings for disease names and synonyms using OpenAI's regular API
and stores them in the mondo_synonym_embeddings table.
"""

import logging
import time
from typing import List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
import numpy as np
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class OpenAIEmbeddingGenerator:
    """
    Generates OpenAI embeddings for MONDO diseases and synonyms using regular API.
    Processes embeddings in configurable batch sizes (default: 1000 per API call).
    """
    
    def __init__(
        self,
        db_connection,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            db_connection: PostgreSQL database connection
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: OpenAI embedding model name
            batch_size: Number of texts to embed per API call (OpenAI allows up to 2048)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        if db_connection is None:
            raise ValueError("Database connection is required")
        
        self.db_connection = db_connection
        self.api_key = api_key or Config.OPEN_API_KEY
        if not self.api_key or self.api_key == "sk-proj-1234567890":
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.batch_size = min(max(1, batch_size), 2048)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.embedding_dim = 1536
        
        logger.info("✓ OpenAI embedding generator initialized with model: %s", self.model_name)
        logger.info(f"✓ Using regular API with batch size: {self.batch_size}")
    
    def _get_embedding_dimension(self) -> int:
        if "large" in self.model_name.lower():
            return 3072
        return 1536
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using regular API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                logger.warning(
                    f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    raise
        
        return []
    
    def generate_all_embeddings(self) -> Tuple[int, int]:
        """
        Generate embeddings for all records in mondo_synonym_embeddings 
        where embedding_built = FALSE using regular OpenAI API.
        
        Returns:
            Tuple of (total_processed, total_errors)
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT mondo_id, disease_name, synonym
                FROM source.mondo_synonym_embeddings
                WHERE embedding_built = FALSE
                ORDER BY mondo_id, synonym
            """)
            
            records = cursor.fetchall()
            total_records = len(records)
            
            if total_records == 0:
                logger.info("No records need embeddings. All embeddings are built.")
                return 0, 0
            
            logger.info(f"Found {total_records} records that need embeddings")
            
            return self._generate_with_regular_api(records)
            
        finally:
            cursor.close()
    
    def _generate_with_regular_api(self, records: List) -> Tuple[int, int]:
        """
        Generate embeddings using regular OpenAI API (one-by-one or in small batches).
        
        Args:
            records: List of database records
            
        Returns:
            Tuple of (total_processed, total_errors)
        """
        logger.info(f"Using regular OpenAI API with batch size: {self.batch_size}")
        
        processed_count = 0
        error_count = 0
        for i in tqdm(range(0, len(records), self.batch_size), desc="Generating embeddings"):
            batch = records[i:i + self.batch_size]
            
            texts = []
            record_keys = []
            
            for record in batch:
                synonym_text = record['synonym'] or record['disease_name']
                texts.append(synonym_text)
                record_keys.append((record['mondo_id'], record['synonym']))
            
            try:
                # Generate embeddings for this batch
                embeddings = self._generate_embeddings_batch(texts)
                
                if len(embeddings) != len(record_keys):
                    logger.warning(f"Embedding count mismatch: got {len(embeddings)}, expected {len(record_keys)}")
                    error_count += abs(len(embeddings) - len(record_keys))
                
                # Update database
                update_cursor = self.db_connection.cursor()
                for idx, ((mondo_id, synonym), embedding) in enumerate(zip(record_keys, embeddings)):
                    try:
                        embedding_str = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                        
                        update_cursor.execute("""
                            UPDATE source.mondo_synonym_embeddings
                            SET embedding = %s::vector,
                                embedding_built = TRUE
                            WHERE mondo_id = %s AND synonym = %s
                        """, (embedding_str, mondo_id, synonym))
                        
                        processed_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error updating embedding for {mondo_id}/{synonym}: {e}")
                
                self.db_connection.commit()
                update_cursor.close()
                if i + self.batch_size < len(records):
                    time.sleep(0.1)
                
            except Exception as e:
                error_count += len(batch)
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {e}")
                self.db_connection.rollback()
        
        logger.info(f"✓ Generated embeddings: {processed_count} successful, {error_count} errors")
        return processed_count, error_count
    
    def generate_embeddings_for_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts (utility method).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._generate_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings

