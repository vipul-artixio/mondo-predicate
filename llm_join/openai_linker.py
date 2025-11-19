"""
OpenAI-based Disease Extractor and Matcher

Extracts diseases from indication text using OpenAI's chat API,
then matches them against MONDO ontology using OpenAI embeddings stored in the database.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
import numpy as np
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class LLMMatchResult:
    """Disease match returned by OpenAILinker."""
    mondo_id: str
    mondo_name: str
    matched_text: str
    similarity: float
    extraction_method: str = "openai"


class OpenAILinker:
    """
    OpenAI-based disease extractor and matcher.
    
    Uses OpenAI for disease extraction and embeddings from database for matching.
    """
    
    def __init__(
        self,
        *,
        db_connection,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-small",
        cache_dir: Optional[str] = None,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        throttle_every: int = 10,
        throttle_sleep: float = 2.0,
    ):
        """
        Initialize OpenAI linker with database connection.
        
        Args:
            db_connection: psycopg2 connection object
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: OpenAI chat model name
            embedding_model_name: OpenAI embedding model name
            cache_dir: Directory to cache results
            batch_size: Number of indications to process per batch
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        if db_connection is None:
            raise ValueError("Database connection is required")
        
        self.api_key = api_key or Config.OPEN_API_KEY
        if not self.api_key or self.api_key == "sk-proj-1234567890":
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.throttle_every = max(0, throttle_every)
        self.throttle_sleep = max(0.0, throttle_sleep)
        self._api_call_count = 0
        self.db_connection = db_connection
        

        cursor = self.db_connection.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM source.mondo_synonym_embeddings
                WHERE embedding_built = TRUE AND embedding IS NOT NULL
            """)
            embedding_count = cursor.fetchone()[0]
            if embedding_count == 0:
                logger.warning("No embeddings found in database. Please run embedding generator first.")
            else:
                logger.info(f"✓ Database contains {embedding_count} embeddings (will use vector similarity search)")
        finally:
            cursor.close()
        
        logger.info("✓ OpenAI linker initialized with model: %s", self.model_name)
    
    
    def _check_extracted_disease(self, product_name: str) -> Optional[str]:
        """
        Check if disease has already been extracted for this product_name.
        
        Args:
            product_name: Product name to check
            
        Returns:
            Extracted disease text if found, None otherwise
        """
        cursor = self.db_connection.cursor()
        try:
            cursor.execute("""
                SELECT extracted_disease
                FROM source.drug_extracted_disease
                WHERE product_name = %s
                LIMIT 1
            """, (product_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
        finally:
            cursor.close()
    
    def _store_extracted_disease(self, product_name: str, extracted_disease: str) -> None:
        """
        Store extracted disease for a product_name.
        
        Args:
            product_name: Product name
            extracted_disease: Extracted disease text (pipe-separated)
        """
        cursor = self.db_connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO source.drug_extracted_disease (product_name, extracted_disease)
                VALUES (%s, %s)
                ON CONFLICT (product_name) DO UPDATE
                SET extracted_disease = EXCLUDED.extracted_disease,
                    updated_at = CURRENT_TIMESTAMP
            """, (product_name, extracted_disease))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing extracted disease: {e}")
            self.db_connection.rollback()
        finally:
            cursor.close()
    
    def _extract_diseases_with_openai(
        self,
        predicate_id: str,
        indication: str,
        product_name: Optional[str] = None,
    ) -> List[str]:
        """
        Extract disease names from indication text using OpenAI.
        
        Args:
            predicate_id: Predicate ID for logging
            indication: Indication text to extract diseases from
            product_name: Product name (for caching)
            
        Returns:
            List of extracted disease names
        """
        # Check if already extracted
        if product_name:
            cached = self._check_extracted_disease(product_name)
            if cached:
                logger.info(f"Using cached extraction for {product_name}")
                diseases = [d.strip() for d in cached.split('|') if d.strip()]
                return diseases
        
        prompt = f"""You are a medical expert. Extract all disease names from the following drug indication text.

            Indication: "{indication}"

            Instructions:
            1. Extract ALL disease names that are explicitly mentioned in the text. Include:
            - Infectious diseases (e.g., Tuberculosis, HIV, Pneumonia)
            - Chronic diseases (e.g., Emphysema, Bronchitis, Cystic Fibrosis)
            - Genetic/hereditary diseases
            - Any other medical conditions that are diseases
            2. EXCLUDE: anatomical parts (e.g., lung, heart), procedures (e.g., surgery, bronchoscopy), symptoms only (e.g., pain, fever)
            3. For each disease, return:
            - The standard medical full term
            - The commonly used short form or abbreviation (if one exists), e.g., HIV → Human Immunodeficiency Virus infection
            4. If a disease has no known short form, return only the full disease name.
            5. List each disease on a separate line in the format:
            Full Name,Short Form
            Example: Human Immunodeficiency Virus Infection,HIV
            6. If multiple diseases are mentioned, list all of them.
            7. If no diseases are found, return "NONE".
            8. Return ONLY the disease names in the required format — no explanations.

            Disease Names:
            """
        
        disease_line_regex = re.compile(r"^(?P<full>[^,]+?)(?:\s*,\s*(?P<short>.+))?$")
        
        for attempt in range(self.max_retries):
            try:
                self._register_api_call()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": """You are a medical expert specialized in extracting disease names from drug indication text.
                        Extract diseases that are explicitly mentioned by name in the text, including:
                        - Primary conditions being treated (e.g., hypertension, diabetes)
                        - Complications or outcomes being prevented (e.g., stroke, myocardial infarction)
                        - Comorbid conditions mentioned (e.g., kidney disease, heart failure)
                        Focus on actual disease names, not symptoms or anatomical parts.
                        """},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=1000,
                    prompt_cache_key="disease_extractor_v1",
                )
                
                diseases = []
                seen_diseases = set()
                
                for raw_line in response.choices[0].message.content.strip().split('\n'):
                    line = raw_line.strip()
                    line = line.lstrip('0123456789.-*• ')
                    if not line or line.upper() == "NONE" or len(line) <= 2:
                        continue

                    match = disease_line_regex.match(line)
                    if match:
                        full_name = match.group('full').strip()
                        short_form = match.group('short')
                        if short_form:
                            short_form = short_form.strip()
                        if full_name:
                            full_name_lower = full_name.lower().strip()
                            if full_name_lower not in seen_diseases:
                                seen_diseases.add(full_name_lower)
                                diseases.append(full_name)
                    else:
                        line_lower = line.lower().strip()
                        if line_lower not in seen_diseases:
                            seen_diseases.add(line_lower)
                            diseases.append(line)
                
                # Store extracted diseases
                if product_name and diseases:
                    extracted_text = '|'.join(diseases)
                    self._store_extracted_disease(product_name, extracted_text)
                
                return diseases
                
            except Exception as e:
                logger.warning(
                    f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}) "
                    f"for predicate {predicate_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to extract diseases for predicate {predicate_id} after {self.max_retries} attempts")
                    return []
        
        return []
    
    def _register_api_call(self) -> None:
        """Track OpenAI API usage and sleep after configurable bursts."""
        self._api_call_count += 1
        if self.throttle_every <= 0:
            return
        if self._api_call_count % self.throttle_every == 0:
            logger.info(
                "OpenAI rate limiting: reached %d calls, sleeping for %.2f seconds",
                self._api_call_count,
                self.throttle_sleep,
            )
            if self.throttle_sleep > 0:
                time.sleep(self.throttle_sleep)
    
    def _generate_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a query text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=[text],
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding
            except Exception as e:
                logger.warning(f"OpenAI embedding error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts")
                    return None
        
        return None
    
    def _match_disease_to_mondo(
        self,
        disease_name: str,
        min_score: float = 0.75,
        top_k: int = 20,
    ) -> List[LLMMatchResult]:
        """
        Match extracted disease name to MONDO ontology using PostgreSQL vector similarity search.
        Uses HNSW index for fast similarity search without loading all embeddings into memory.
        
        Args:
            disease_name: Disease name extracted by OpenAI
            min_score: Minimum similarity threshold (cosine similarity)
            top_k: Maximum number of matches to return
            
        Returns:
            List of matched MONDO concepts with confidence scores
        """
        matches: List[LLMMatchResult] = []
        
        # Generate embedding for the query
        query_embedding = self._generate_query_embedding(disease_name)
        if query_embedding is None:
            logger.warning(f"Failed to generate embedding for '{disease_name}'")
            return matches

        embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'
        max_distance = 1.0 - min_score
        
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                SELECT 
                    mondo_id,
                    disease_name,
                    synonym,
                    1 - (embedding <=> %s::vector) as similarity
                FROM source.mondo_synonym_embeddings
                WHERE embedding_built = TRUE
                  AND embedding IS NOT NULL
                  AND (embedding <=> %s::vector) <= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, max_distance, embedding_str, top_k * 5))
            
            seen_mondo_ids = set()
            for row in cursor.fetchall():
                similarity = float(row['similarity'])
                mondo_id = row['mondo_id']
                if mondo_id in seen_mondo_ids:
                    continue
                
                if similarity >= min_score:
                    seen_mondo_ids.add(mondo_id)
                    matches.append(LLMMatchResult(
                        mondo_id=mondo_id,
                        mondo_name=row['disease_name'],
                        matched_text=disease_name,
                        similarity=similarity,
                        extraction_method="openai_embedding"
                    ))
                    
                    if len(matches) >= top_k:
                        break
        
        except Exception as e:
            logger.error(f"Error during vector similarity search: {e}")
        finally:
            cursor.close()
        
        return matches
    
    def match(
        self,
        text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.75,
        context: Optional[str] = None,
        predicate_id: Optional[str] = None,
        product_name: Optional[str] = None,
    ) -> List[LLMMatchResult]:
        """
        Extract diseases from indication text and match to MONDO.
        
        Args:
            text: Indication text (not used directly, context is used)
            top_k: Number of top matches to return
            min_score: Minimum similarity threshold
            context: Full indication text for OpenAI extraction
            predicate_id: Predicate ID for logging
            product_name: Product name for caching
            
        Returns:
            List of matched MONDO concepts
        """
        if not context:
            context = text
        
        if not predicate_id:
            predicate_id = "unknown"
        
        extracted_diseases = self._extract_diseases_with_openai(
            predicate_id, 
            context,
            product_name
        )
        
        logger.info(f"OpenAI extracted {len(extracted_diseases)} disease names for predicate {predicate_id}: {extracted_diseases}")
        
        if not extracted_diseases:
            logger.warning(f"No diseases extracted by OpenAI for predicate {predicate_id}")
            return []
        
        all_matches: List[LLMMatchResult] = []
        for disease_name in extracted_diseases:
            matches = self._match_disease_to_mondo(disease_name, min_score)
            if matches:
                logger.info(f"  '{disease_name}' matched to {len(matches)} MONDO IDs: {[m.mondo_id for m in matches[:3]]}")
            else:
                logger.warning(f"  '{disease_name}' - NO MONDO MATCH (min_score={min_score})")
            all_matches.extend(matches)
        
        seen_ids = set()
        unique_matches = []
        for match in sorted(all_matches, key=lambda x: x.similarity, reverse=True):
            if match.mondo_id not in seen_ids:
                seen_ids.add(match.mondo_id)
                unique_matches.append(match)
        
        if top_k and top_k > 0:
            return unique_matches[:top_k]
        return unique_matches

