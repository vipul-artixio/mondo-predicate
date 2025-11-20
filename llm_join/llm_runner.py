"""
LLM Disease Linking Runner

Main entry point for the LLM-based disease entity linking pipeline.
Uses OpenAI to extract diseases from indications and embeddings for matching.
"""

import logging
import os
import sys
import psycopg2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from llm_disease_linker import LLMDiseaseLinkingPipeline
from openai_embedding_generator import OpenAIEmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Create and return a database connection using Config settings.
    
    Returns:
        psycopg2 connection object
    """
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def check_prerequisites(connection):
    """
    Check if required tables and data exist before running the pipeline.
    
    Args:
        connection: Database connection
        
    Returns:
        bool: True if prerequisites are met, False otherwise
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM source.mondo_tsv_data
        """)
        mondo_count = cursor.fetchone()[0]
        
        if mondo_count == 0:
            logger.error("source.mondo_tsv_data table is empty. Please run mondo_sync first.")
            cursor.close()
            return False
        
        logger.info(f"Found {mondo_count} MONDO diseases in database")
        
        cursor.execute("""
            SELECT COUNT(DISTINCT product_name) 
            FROM drug.drug_predicate_assessments
            WHERE indication IS NOT NULL 
              AND TRIM(indication) != ''
              AND LOWER(indication) != 'null'
        """)
        drug_count = cursor.fetchone()[0]
        
        if drug_count == 0:
            logger.error("drug.drug_predicate_assessments table has no valid indications.")
            cursor.close()
            return False
        
        logger.info(f"Found {drug_count} unique products with valid indications")
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'source' 
                AND table_name = 'indication_and_disease'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.error("source.indication_and_disease table does not exist. Please run schema.sql first.")
            cursor.close()
            return False
        
        # Check if embeddings are built
        cursor.execute("""
            SELECT COUNT(*) FROM source.mondo_synonym_embeddings
            WHERE embedding_built = TRUE AND embedding IS NOT NULL
        """)
        embedding_count = cursor.fetchone()[0]
        
        if embedding_count == 0:
            logger.warning(
                "No embeddings found in mondo_synonym_embeddings. "
                "Please run the embedding generator first."
            )
        else:
            logger.info(f"✓ Found {embedding_count} embeddings in database")
        
        openai_api_key = os.getenv("OPENAI_API_KEY") or Config.OPEN_API_KEY
        if not openai_api_key or openai_api_key == "sk-proj-1234567890":
            logger.error(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it to use the OpenAI-based disease extraction."
            )
            cursor.close()
            return False
        
        logger.info("✓ OpenAI API key found")
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"Error checking prerequisites: {e}")
        return False


def main():
    """
    Main entry point for LLM-based disease linking process.
    """
    connection = None
    
    try:
        logger.info("=" * 80)
        logger.info("LLM-based Disease Entity Linking Pipeline (OpenAI)")
        logger.info("=" * 80)
        connection = get_db_connection()
        if not check_prerequisites(connection):
            logger.error("Prerequisites not met. Exiting.")
            return 1
        
        # Generate embeddings for records where embedding_built = FALSE
        logger.info("=" * 80)
        logger.info("Checking for missing embeddings...")
        logger.info("=" * 80)
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM source.mondo_synonym_embeddings
            WHERE embedding_built = FALSE
        """)
        missing_count = cursor.fetchone()[0]
        cursor.close()
        
        if missing_count > 0:
            logger.info(f"Found {missing_count} records with embedding_built = FALSE")
            logger.info("Generating embeddings...")
            
            try:
                generator = OpenAIEmbeddingGenerator(
                    db_connection=connection,
                    api_key=Config.OPEN_API_KEY,
                    model_name=Config.OPENAI_EMBEDDING_MODEL_NAME,
                    batch_size=1000,
                )
                processed, errors = generator.generate_all_embeddings()
                logger.info(f"✓ Generated {processed} embeddings ({errors} errors)")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                logger.warning("Continuing with existing embeddings...")
        else:
            logger.info("✓ All embeddings are already built")
        
        logger.info("=" * 80)
        logger.info("Starting Disease Linking Pipeline...")
        logger.info("=" * 80)
        
        pipeline = LLMDiseaseLinkingPipeline(
            connection=connection,
            openai_api_key=Config.OPEN_API_KEY,
            openai_model_name=Config.OPEN_AI_MODEL,
            embedding_model_name=Config.OPENAI_EMBEDDING_MODEL_NAME,
            min_confidence=Config.LLM_MIN_CONFIDENCE,
            llm_top_k=Config.LLM_TOP_K,
            llm_threshold=Config.LLM_THRESHOLD,
        )
        
        stats = pipeline.run_pipeline()
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
        
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main())
