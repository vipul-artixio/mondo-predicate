import logging
import sys
import psycopg2
from config import Config
from mondo_sync import MondoSync

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Create and return a database connection using Config settings.
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
    Main entry point for Mondo sync process.
    Downloads mondo_nodes.tsv and inserts biolink:Disease records into the database.
    """
    connection = None
    
    try:
        logger.info("=" * 80)
        logger.info("Starting Mondo Data Sync")
        logger.info("=" * 80)
        
        # Establish database connection
        connection = get_db_connection()
        
        # Initialize and run sync
        mondo_sync = MondoSync(connection)
        record_count = mondo_sync.sync()
        
        logger.info("=" * 80)
        logger.info(f"Mondo sync completed successfully!")
        logger.info(f"Total Disease records processed: {record_count}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Mondo sync failed: {e}", exc_info=True)
        return 1
        
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main())
