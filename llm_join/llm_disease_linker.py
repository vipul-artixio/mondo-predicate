"""
LLM-based Disease Entity Linking Module

This module links drug indication text to MONDO disease IDs using Groq LLM extraction
instead of scispaCy NER. The rest of the pipeline (ensemble matching, validation, etc.)
remains the same as the disease_join module.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor

from openai_linker import OpenAILinker, LLMMatchResult
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DiseaseMatch:
    """Represents a matched disease with MONDO ID."""
    mondo_id: str
    disease_name: str
    matched_phrase: str
    match_type: str 
    confidence: float
    extraction_source: str  # 'llm'
    similarity: Optional[float] = None


class DiseaseValidator:
    """
    Rule-based validator for disease entities.
    Implements filtering criteria:
    - Reject generic terms (neoplasm, cancer NOS)
    - Reject obsolete MONDO terms
    - Reject phenotype & process terms
    """
    
    # Generic terms to reject
    GENERIC_TERMS = {
        'neoplasm', 'neoplasms', 'cancer', 'tumor', 'tumors', 'tumour', 'tumours',
        'disease', 'disorder', 'syndrome', 'condition', 'malignancy', 'malignancies',
        'carcinoma', 'sarcoma', 'infection', 'inflammation', 'injury',
        'cancer nos', 'carcinoma nos', 'sarcoma nos', 'neoplasm nos',
    }
    
    # Phenotype-related terms to reject
    PHENOTYPE_TERMS = {
        'symptom', 'symptoms', 'sign', 'signs', 'feature', 'features',
        'manifestation', 'manifestations', 'presentation', 'presentations',
        'abnormality', 'abnormalities', 'defect', 'defects',
    }
    
    # Process-related terms to reject
    PROCESS_TERMS = {
        'process', 'processes', 'pathway', 'pathways', 'mechanism', 'mechanisms',
        'response', 'responses', 'reaction', 'reactions', 'metabolism',
    }
    
    def __init__(self, connection):
        self.connection = connection
        self._load_mondo_metadata()
    
    def _load_mondo_metadata(self):
        """Load MONDO metadata for validation (obsolete terms, phenotypes, etc.)"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT DISTINCT mondo_id
                FROM source.mondo_tsv_data
                WHERE disease_name ILIKE '%obsolete%'
                   OR disease_name ILIKE '%deprecated%'
            """)
            self.obsolete_ids = {row['mondo_id'] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT DISTINCT mondo_id, disease_name
                FROM source.mondo_tsv_data
                WHERE disease_name ILIKE '%phenotype%'
                   OR disease_name ILIKE '%process%'
                   OR disease_name ILIKE '%trait%'
            """)
            self.phenotype_process_ids = {row['mondo_id'] for row in cursor.fetchall()}
            
            cursor.close()
            logger.info(f"✓ Loaded {len(self.obsolete_ids)} obsolete MONDO terms")
            logger.info(f"✓ Loaded {len(self.phenotype_process_ids)} phenotype/process terms")
            
        except Exception as e:
            logger.warning(f"Could not load MONDO metadata: {e}")
            self.obsolete_ids = set()
            self.phenotype_process_ids = set()
    
    def is_generic_term(self, text: str) -> bool:
        """Check if the extracted text is a generic term."""
        normalized = text.lower().strip()
        
        if len(normalized) <= 2:
            return True
        
        words = normalized.split()
        if len(words) == 1 and len(normalized) <= 3:
            return True
        
        if normalized in self.GENERIC_TERMS:
            return True
        
        if len(words) == 1 and words[0] in self.GENERIC_TERMS:
            return True
        
        if 'nos' in normalized or 'not otherwise specified' in normalized:
            return True
        
        return False
    
    def is_phenotype_or_process(self, text: str, mondo_id: Optional[str] = None) -> bool:
        """Check if the term is a phenotype or process rather than a disease."""
        normalized = text.lower().strip()
        
        for term in self.PHENOTYPE_TERMS | self.PROCESS_TERMS:
            if term in normalized:
                return True
        
        if mondo_id and mondo_id in self.phenotype_process_ids:
            return True
        
        return False
    
    def is_obsolete(self, mondo_id: str) -> bool:
        """Check if the MONDO ID is obsolete."""
        return mondo_id in self.obsolete_ids
    
    def validate_disease(self, text: str, mondo_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate a disease entity against all rules.
        
        Returns:
            Tuple[bool, str]: (is_valid, rejection_reason)
        """
        if self.is_generic_term(text):
            return False, "generic_term"
        
        if self.is_phenotype_or_process(text, mondo_id):
            return False, "phenotype_or_process"
        
        if mondo_id and self.is_obsolete(mondo_id):
            return False, "obsolete_mondo"
        
        return True, "valid"


class LLMDiseaseLinkingPipeline:
    """Main pipeline for linking drug indications to MONDO disease IDs using OpenAI LLM."""
    
    def __init__(
        self,
        connection,
        openai_api_key: Optional[str] = None,
        openai_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        min_confidence: float = 0.75,
        llm_top_k: int = 5,
        llm_threshold: float = 0.75,
    ):
        """
        Initialize the LLM-based disease linking pipeline.
        
        Args:
            connection: PostgreSQL database connection
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            openai_model_name: OpenAI chat model name (defaults to Config.OPEN_AI_MODEL)
            embedding_model_name: OpenAI embedding model name (defaults to Config.OPENAI_EMBEDDING_MODEL_NAME)
            min_confidence: Minimum combined confidence to accept a match (0-1)
            llm_top_k: Number of top LLM matches to consider
            llm_threshold: Minimum similarity score for LLM matches (cosine similarity)
        """
        self.connection = connection
        self.min_confidence = min_confidence
        self.llm_top_k = llm_top_k
        self.llm_threshold = llm_threshold
        
        self.validator = DiseaseValidator(connection)
        logger.info("✓ Rule-based disease validator initialized")
        
        self.llm_linker: Optional[OpenAILinker] = None
        try:
            self.llm_linker = OpenAILinker(
                db_connection=connection,
                api_key=openai_api_key,
                model_name=openai_model_name or Config.OPEN_AI_MODEL,
                embedding_model_name=embedding_model_name or Config.OPENAI_EMBEDDING_MODEL_NAME,
                throttle_every=int(getattr(Config, "OPENAI_THROTTLE_EVERY", 10)),
                throttle_sleep=float(getattr(Config, "OPENAI_THROTTLE_SLEEP", 2.0)),
            )
            logger.info("✓ OpenAI LLM linker initialized with model: %s", openai_model_name or Config.OPEN_AI_MODEL)
        except Exception as exc:
            logger.error("Failed to initialize OpenAI LLM linker: %s", exc)
            raise
        
        logger.info("LLM disease linking pipeline initialized with OpenAI extraction and embeddings")
    
    def _convert_llm_to_disease_match(self, llm_match: LLMMatchResult) -> DiseaseMatch:
        """Convert LLMMatchResult to DiseaseMatch."""
        return DiseaseMatch(
            mondo_id=llm_match.mondo_id,
            disease_name=llm_match.mondo_name,
            matched_phrase=llm_match.matched_text,
            match_type=llm_match.extraction_method,
            confidence=llm_match.similarity,
            extraction_source="llm",
            similarity=llm_match.similarity,
        )
    
    def link_indication(
        self,
        predicate_id: str,
        indication: str,
        product_name: Optional[str] = None,
    ) -> List[DiseaseMatch]:
        """
        Link a single drug indication to MONDO disease IDs using OpenAI LLM.
        
        Args:
            predicate_id: Predicate ID for logging
            indication: Indication text to extract diseases from
            product_name: Product name for caching extracted diseases
            
        Returns:
            List of DiseaseMatch objects
        """
        if not indication or not indication.strip():
            return []
        
        matches: List[DiseaseMatch] = []
        seen_mondo_ids: Set[str] = set()
        
        logger.info(f"Processing predicate {predicate_id}: {indication[:100]}...")
        
        llm_matches = self.llm_linker.match(
            text=indication,
            top_k=self.llm_top_k,
            min_score=self.llm_threshold,
            context=indication,
            predicate_id=predicate_id,
            product_name=product_name,
        )
        
        logger.info(f"LLM extracted {len(llm_matches)} matches for predicate {predicate_id}")
        for llm_match in llm_matches:
            logger.info(f"  - {llm_match.matched_text} -> {llm_match.mondo_id} ({llm_match.mondo_name}) [score: {llm_match.similarity:.3f}]")
        
        for llm_match in llm_matches:
            is_valid, reason = self.validator.validate_disease(
                llm_match.matched_text,
                llm_match.mondo_id
            )
            
            if not is_valid:
                logger.debug(
                    f"Rejected LLM match '{llm_match.mondo_id}' for '{llm_match.matched_text}': {reason}"
                )
                continue
            
            if llm_match.mondo_id not in seen_mondo_ids:
                disease_match = self._convert_llm_to_disease_match(llm_match)
                matches.append(disease_match)
                seen_mondo_ids.add(llm_match.mondo_id)
        
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return matches
    
    def run_pipeline(self) -> Dict[str, int]:
        """
        Run the complete disease linking pipeline on all drug indications.
        Groups by product_name to process each unique indication once,
        then stores results for all associated predicate_ids.
        
        Returns:
            Statistics dictionary with counts
        """
        logger.info("=" * 80)
        logger.info("Starting LLM-based Disease Linking Pipeline")
        logger.info("=" * 80)
        
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            WITH valid_predicates AS (
                SELECT
                    id,
                    product_name,
                    indication,
                    updated_at
                FROM drug.drug_predicate_assessments
                WHERE indication IS NOT NULL
                  AND TRIM(indication) != ''
                  AND LOWER(indication) != 'null'
            ), ranked_indications AS (
                SELECT
                    id,
                    product_name,
                    indication,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_name
                        ORDER BY updated_at DESC NULLS LAST, id DESC
                    ) AS row_rank
                FROM valid_predicates
            ), product_predicates AS (
                SELECT
                    product_name,
                    ARRAY_AGG(DISTINCT id ORDER BY id) AS predicate_ids
                FROM valid_predicates
                GROUP BY product_name
            )
            SELECT
                r.product_name,
                r.indication,
                p.predicate_ids
            FROM ranked_indications r
            JOIN product_predicates p
              ON p.product_name = r.product_name
            WHERE r.row_rank = 1
            ORDER BY r.product_name
        """)
        
        records = cursor.fetchall()
        cursor.close()
        
        total_predicates = sum(len(r['predicate_ids']) for r in records)
        logger.info(f"Found {len(records)} products with valid indications")
        logger.info(f"Total predicate_ids represented: {total_predicates}")
        
        stats = {
            'total_processed': 0,
            'total_matches': 0,
            'records_with_matches': 0,
            'records_without_matches': 0,
        }
        logger.info("Starting disease linking (incremental update mode)...")
        
        batch_size = 10
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            for record in batch:
                product_name = record['product_name']
                indication = record['indication']
                predicate_ids = record['predicate_ids']
                
                primary_predicate_id = predicate_ids[0] if predicate_ids else 'unknown'
                
                matches = self.link_indication(primary_predicate_id, indication, product_name)
                
                stats['total_processed'] += 1
                stats['total_matches'] += len(matches) * len(predicate_ids)
                
                if matches:
                    stats['records_with_matches'] += 1
                    
                    new_mondo_ids = {match.mondo_id for match in matches}
                    
                    cursor = self.connection.cursor()
                    for predicate_id in predicate_ids:
                        cursor.execute("""
                            SELECT mondo_disease_ids
                            FROM source.indication_and_disease
                            WHERE predicate_id = %s
                        """, (predicate_id,))
                        
                        existing_row = cursor.fetchone()
                        if existing_row and existing_row[0]:
                            existing_ids = set(existing_row[0].split(',')) if existing_row[0] else set()
                            merged_ids = existing_ids | new_mondo_ids
                            mondo_ids_str = ','.join(sorted(merged_ids))
                            logger.debug(
                                f"Merging matches for predicate {predicate_id}: "
                                f"existing={len(existing_ids)}, new={len(new_mondo_ids)}, merged={len(merged_ids)}"
                            )
                        else:
                            mondo_ids_str = ','.join(sorted(new_mondo_ids))
                        
                        cursor.execute("""
                            INSERT INTO source.indication_and_disease 
                            (predicate_id, mondo_disease_ids)
                            VALUES (%s, %s)
                            ON CONFLICT (predicate_id) DO UPDATE
                            SET mondo_disease_ids = EXCLUDED.mondo_disease_ids,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            predicate_id,
                            mondo_ids_str,
                        ))
                    self.connection.commit()
                    cursor.close()
                    
                    mondo_ids_str = ','.join(sorted(new_mondo_ids))
                    
                    logger.info(
                        f"✓ {product_name} ({len(predicate_ids)} predicates): Found {len(matches)} matches - "
                        f"{mondo_ids_str}"
                    )
                else:
                    stats['records_without_matches'] += 1
                    logger.info(f"✗ {product_name}: No matches found")
            
            logger.info(
                f"Progress: {min(i + batch_size, len(records))}/{len(records)} products "
                f"({100 * min(i + batch_size, len(records)) / len(records):.1f}%)"
            )
        
        logger.info("=" * 80)
        logger.info("Pipeline Statistics")
        logger.info("=" * 80)
        logger.info(f"Total indications processed: {stats['total_processed']}")
        logger.info(f"Records with matches: {stats['records_with_matches']}")
        logger.info(f"Records without matches: {stats['records_without_matches']}")
        logger.info(f"Total MONDO matches: {stats['total_matches']}")
        if stats['records_with_matches'] > 0:
            logger.info(
                f"Average matches per record: "
                f"{stats['total_matches'] / stats['records_with_matches']:.2f}"
            )
        logger.info("=" * 80)
        
        return stats


def demo() -> None:
    """Quick manual demo for local testing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import Config
    
    connection = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        pipeline = LLMDiseaseLinkingPipeline(
            connection=connection,
            gemini_model_name="gemini-1.5-flash",
            min_confidence=0.7,
            llm_top_k=5,
            llm_threshold=0.7,
        )
        
        test_cases = [
            ("pred_001", "Treatment of HIV-1 infection in adults and pediatric patients"),
            ("pred_002", "Management of type 2 diabetes mellitus in adults"),
            ("pred_003", "Prevention of cardiovascular disease in high-risk patients"),
        ]
        
        for predicate_id, indication in test_cases:
            print(f"\n{'='*80}")
            print(f"Predicate: {predicate_id}")
            print(f"Indication: {indication}")
            print(f"{'='*80}")
            
            matches = pipeline.link_indication(predicate_id, indication)
            
            if matches:
                print(f"Found {len(matches)} matches:")
                for match in matches:
                    print(f"  → {match.mondo_id}: {match.disease_name}")
                    print(f"    Matched: '{match.matched_phrase}' (confidence: {match.confidence:.3f})")
                    print(f"    Type: {match.match_type}")
            else:
                print("  No matches found")
    
    finally:
        connection.close()


if __name__ == "__main__":  # pragma: no cover
    demo()
