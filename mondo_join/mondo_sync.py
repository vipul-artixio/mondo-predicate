import logging
from datetime import datetime
from typing import Optional
import csv
from io import StringIO

import psycopg2
import psycopg2.extras
import requests

from config import Config


logger = logging.getLogger(__name__)


class MondoSync:
    # Animal species to filter out from disease names
    # Note: Order matters - longer/more specific names should be checked first
    ANIMAL_SPECIES = set([
        # Common mammals (with variants)
        "dog", "cat", "horse", "cow", "goat", "sheep", "pig",
        "camel", "alpaca", "llama", "buffalo", "yak", "non human", "non human animal", "non-human animal",
        "domestic cat", "domestic dog", "domestic pig",
        "Arabian camel", "water buffalo", "cattle", "deer", "ass","guanaco","monkey","leaf-monkey","langur","lutung","springbok","hawk",
        # Rodents
        "mouse", "mice", "rat", "hamster", "gerbil", "chinchilla","marmoset","panda","wapiti","fishes","markhor","rhinoceros","elephant","shrew","mallard",
        "golden hamster", "siberian hamster", "campbell's hamster","elk","coati","tetra","kudu","degu","Kaka","capuchin","lovebird","racoon","bongo","cockatiel",
        "degu", "guinea pig", "domestic guinea pig", "vole","rabbit",
        # Primates
        "rhesus macaque", "macaque", "baboon", "gibbon", "chimpanzee",
        "gorilla", "red shanked douc langur", "francois's langur",
        # Carnivores
        "tiger", "lion", "leopard", "puma", "lynx", "cheetah", "otter",
        "mink", "ferret", "fox", "coyote", "wolf",
        # Birds
        "chicken", "duck", "goose", "turkey", "quail", "pigeon",
        "owl", "great horned owl", "falcon", "eagle", "ostrich",
        "emu", "parrot", "budgerigar", "macaw", "sparrow",
        "Japanese quail",
        # Fish
        "zebrafish", "tilapia", "salmon", "trout", "catfish",
        "carp", "medaka", "goldfish","tench",
        # Reptiles / Amphibians
        "turtle", "tortoise", "lizard", "snake", "python",
        "gecko", "iguana", "frog", "toad",
        # Exotic/others
        "kangaroo", "koala", "wallaby", "sloth", "anteater",
        "squirrel", "chipmunk", "bat","zebra","bear",
        # Marine mammals
        "dolphin", "whale", "seal", "sea otter","hedgehogs",
    ])
    
    def __init__(self, connection: psycopg2.extensions.connection):
        self.connection = connection
        self.mondo_base_url = Config.MONDO_BASE_URL

    def download_mondo_nodes(self) -> str:
        """
        Download the mondo_nodes.tsv file from the mondo_base_url.
        """
        logger.info("Downloading mondo_nodes.tsv file from %s", self.mondo_base_url)
        response = requests.get(self.mondo_base_url, timeout=60)
        response.raise_for_status()
        logger.info("Successfully downloaded mondo_nodes.tsv")
        return response.text

    def _contains_animal_word(self, text: str) -> bool:
        """
        Check if the text contains any animal species (simple substring matching).
        Detects animal-related diseases by checking if any animal name appears in the text.
        
        Args:
            text: Text to check (disease name)
            
        Returns:
            True if text contains any animal species, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Sort animals by length (longest first) to match multi-word animals before single words
        # This ensures "domestic cat" matches before just "cat"
        sorted_animals = sorted(self.ANIMAL_SPECIES, key=len, reverse=True)
        
        # Simple substring check - if animal name appears anywhere in the text
        for animal in sorted_animals:
            if animal.lower() in text_lower:
                return True
        
        return False
    
    def _normalize_mondo_id(self, mondo_id: str) -> str:
        """
        Normalize mondo_id from MONDO:0000101 format to MONDO_0000101 format.
        Replaces colon with underscore.
        
        Args:
            mondo_id: Original mondo_id (e.g., "MONDO:0000101")
            
        Returns:
            Normalized mondo_id (e.g., "MONDO_0000101")
        """
        if not mondo_id:
            return mondo_id
        return mondo_id.replace(":", "_")
    
    def process_mondo_nodes(self, tsv_content: str) -> int:
        """
        Process the downloaded mondo_nodes.tsv content and store it in the database.
        Only stores records where category (column B) is 'biolink:Disease'.
        Filters out records with animal-related words in disease_name.
        Stores columns: A (id), C (name), D (description), G (synonym)
        Transforms mondo_id from MONDO:0000101 to MONDO_0000101 format.
        
        Returns:
            Number of records inserted/updated
        """
        cursor = self.connection.cursor()
        tsv_reader = csv.reader(StringIO(tsv_content), delimiter='\t')
        header = next(tsv_reader, None)
        if not header:
            logger.warning("Empty TSV file")
            return 0
        
        logger.info(f"TSV Header: {header[:10]}...")
        
        inserted_count = 0
        skipped_count = 0
        animal_filtered_count = 0
        error_count = 0
        embedding_records_count = 0
        batch_size = 100
        for row_num, row in enumerate(tsv_reader, start=2):
            if not row or len(row) < 2:
                continue
            mondo_id_raw = row[0].strip() if len(row) > 0 else None
            category = row[1].strip() if len(row) > 1 else None
            disease_name = row[2].strip() if len(row) > 2 else None
            description = row[3].strip() if len(row) > 3 else None
            synonym = row[6].strip() if len(row) > 6 else None

            if category != "biolink:Disease":
                skipped_count += 1
                continue
            
            if not mondo_id_raw:
                logger.warning(f"Row {row_num}: Missing mondo_id, skipping")
                continue

            if not disease_name or not disease_name.strip():
                skipped_count += 1
                if skipped_count <= 10:  
                    logger.debug(f"Row {row_num}: Skipping record with empty disease_name (mondo_id={mondo_id_raw})")
                continue
            
            mondo_id = self._normalize_mondo_id(mondo_id_raw)
            
            if self._contains_animal_word(disease_name):
                animal_filtered_count += 1
                matched_animals = []
                text_lower = disease_name.lower()
                sorted_animals = sorted(self.ANIMAL_SPECIES, key=len, reverse=True)
                for animal in sorted_animals:
                    if animal.lower() in text_lower:
                        matched_animals.append(animal)
                        break
                logger.info(f"Row {row_num}: Filtered out animal-related disease: '{disease_name}' (matched animal: {matched_animals[0] if matched_animals else 'unknown'})")
                continue
            
            try:
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO source.mondo_tsv_data 
                        (mondo_id, disease_name, description, synonym, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (mondo_id) 
                    DO UPDATE SET 
                        disease_name = EXCLUDED.disease_name,
                        description = EXCLUDED.description,
                        synonym = CASE 
                            WHEN EXCLUDED.synonym IS NOT NULL AND EXCLUDED.synonym != '' 
                            THEN EXCLUDED.synonym 
                            ELSE source.mondo_tsv_data.synonym 
                        END,
                        updated_at = EXCLUDED.updated_at
                """, (
                    mondo_id, 
                    disease_name, 
                    description, 
                    synonym,
                    now,
                    now 
                ))
                
                # Insert into mondo_synonym_embeddings table
                # Insert disease_name as a synonym entry (so disease_name itself is searchable)
                cursor.execute("""
                    INSERT INTO source.mondo_synonym_embeddings 
                        (mondo_id, disease_name, synonym, embedding, embedding_built)
                    VALUES (%s, %s, %s, NULL, FALSE)
                    ON CONFLICT (mondo_id, synonym) 
                    DO UPDATE SET 
                        disease_name = EXCLUDED.disease_name
                """, (mondo_id, disease_name, disease_name))
                embedding_records_count += 1
                
                # Insert each synonym from pipe-separated list
                if synonym and synonym.strip():
                    synonyms_list = [s.strip() for s in synonym.split('|') if s.strip()]
                    for syn in synonyms_list:
                        cursor.execute("""
                            INSERT INTO source.mondo_synonym_embeddings 
                                (mondo_id, disease_name, synonym, embedding, embedding_built)
                            VALUES (%s, %s, %s, NULL, FALSE)
                            ON CONFLICT (mondo_id, synonym) 
                            DO UPDATE SET 
                                disease_name = EXCLUDED.disease_name
                        """, (mondo_id, disease_name, syn))
                        embedding_records_count += 1
                
                inserted_count += 1
                if inserted_count % batch_size == 0:
                    self.connection.commit()
                    logger.info(f"Processed {inserted_count} disease records...")
                    
            except Exception as e:
                self.connection.rollback()
                error_count += 1
                logger.error(f"Error inserting row {row_num} (mondo_id={mondo_id}): {e}")
                if error_count <= 5:
                    logger.error(f"  Data: name={disease_name}, desc={description[:50] if description else None}")
                continue
        
        self.connection.commit()
        cursor.close()
        
        logger.info(f"Total records processed: {inserted_count}")
        logger.info(f"Total records skipped (non-Disease): {skipped_count}")
        logger.info(f"Total records filtered (animal-related): {animal_filtered_count}")
        logger.info(f"Total embedding records created: {embedding_records_count}")
        if error_count > 0:
            logger.warning(f"Total errors encountered: {error_count}")
        
        return inserted_count

    def sync(self) -> int:
        """
        Main sync method: download and process mondo nodes.
        
        Returns:
            Number of records inserted/updated
        """
        try:
            logger.info("Starting Mondo sync...")
            tsv_content = self.download_mondo_nodes()
            count = self.process_mondo_nodes(tsv_content)
            logger.info(f"Mondo sync completed successfully. {count} records processed.")
            return count
        except Exception as e:
            logger.error(f"Error during Mondo sync: {e}", exc_info=True)
            raise