"""
OpenAI Embedding Generator Runner

Generates OpenAI embeddings for all diseases and synonyms in mondo_synonym_embeddings table.
"""

import logging
import sys
import psycopg2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
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


def main():
    """
    Main entry point for embedding generation process.
    """
    connection = None
    
    try:
        logger.info("=" * 80)
        logger.info("OpenAI Embedding Generator")
        logger.info("=" * 80)
        
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'source' 
                AND table_name = 'mondo_synonym_embeddings'
            )
        """)
        table_exists = cursor.fetchone()[0]
        cursor.close()
        
        if not table_exists:
            logger.error("source.mondo_synonym_embeddings table does not exist. Please run schema.sql first.")
            return 1
        
        # Check API key
        api_key = Config.OPEN_API_KEY
        if not api_key or api_key == "sk-proj-1234567890":
            logger.error(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it to generate embeddings."
            )
            return 1
        
        logger.info("✓ OpenAI API key found")
        logger.info(f"✓ Using embedding model: {Config.OPENAI_EMBEDDING_MODEL_NAME}")
        generator = OpenAIEmbeddingGenerator(
            db_connection=connection,
            api_key=api_key,
            model_name=Config.OPENAI_EMBEDDING_MODEL_NAME,
            batch_size=100,
        )
        processed, errors = generator.generate_all_embeddings()
        
        logger.info("=" * 80)
        logger.info("Embedding generation completed!")
        logger.info(f"Processed: {processed} records")
        logger.info(f"Errors: {errors} records")
        logger.info("=" * 80)
        
        return 0 if errors == 0 else 1
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        return 1
        
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main())

