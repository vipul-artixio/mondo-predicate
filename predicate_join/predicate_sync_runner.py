import argparse
import logging
import sys
from datetime import datetime
from typing import Optional
import psycopg2
import psycopg2.extras
from config import Config
from predicate_sync import sync_all_predicate_assessments


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild drug predicate assessments from usa_drug_data and labels.",
    )
    parser.add_argument(
        "--updated-since",
        dest="updated_since",
        help="Only sync records with usa_drug_data.updated_at >= ISO timestamp (UTC).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of usa_drug_data ids to process per transaction (default: 500).",
    )
    return parser.parse_args()


def _parse_updated_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"Invalid --updated-since value '{value}'. Expected ISO format, e.g. 2024-01-31T12:34:56",
        )


def run(updated_since: Optional[datetime] = None, batch_size: int = 500) -> int:
    """
    Execute a predicate sync run and return process exit code (0 success, 1 failure).
    """
    # Only configure logging if not already configured (when run standalone)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
        )
    except Exception as exc:
        logger.error("Failed to connect to database: %s", exc, exc_info=True)
        return 1

    try:
        total = sync_all_predicate_assessments(
            conn,
            updated_since=updated_since,
            batch_size=batch_size,
        )
        logger.info("Predicate sync completed - records processed: %s", total)
        return 0
    except Exception as exc:
        logger.error("Predicate sync failed: %s", exc, exc_info=True)
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            logger.warning("Failed to close database connection cleanly", exc_info=True)


def main(updated_since: Optional[str] = None, batch_size: int = 500) -> int:
    """
    Main entry point that can be called programmatically or from command line.
    
    Args:
        updated_since: ISO timestamp string (optional)
        batch_size: Number of records to process per batch
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    updated_since_dt = _parse_updated_since(updated_since)
    return run(updated_since=updated_since_dt, batch_size=batch_size)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(updated_since=args.updated_since, batch_size=args.batch_size))

