# LLM Join - Disease Entity Linking Pipeline

## Overview

The LLM Join module links drug indication text to MONDO disease ontology IDs using OpenAI's language models for disease extraction and semantic embeddings for matching. This pipeline processes drug predicate assessments and identifies the diseases mentioned in their indications.

## Architecture

The pipeline consists of several key components:

1. **OpenAI Disease Extractor** - Uses GPT models to extract disease names from indication text
2. **Embedding Generator** - Creates vector embeddings for MONDO diseases and synonyms
3. **Vector Similarity Matcher** - Matches extracted diseases to MONDO IDs using cosine similarity
4. **Disease Validator** - Rule-based filtering to reject invalid matches
5. **Ensemble Matcher** (Optional) - Combines multiple matching strategies for improved accuracy
6. **Cross-Encoder Reranker** (Optional) - Reranks matches using a cross-encoder model

## Complete Process Flow

### Step 1: Prerequisites Check (`llm_runner.py`)

Before running the pipeline, the system checks:

- ✅ MONDO disease data exists in `source.mondo_tsv_data`
- ✅ Drug predicate assessments exist in `drug.drug_predicate_assessments`
- ✅ Output table `source.indication_and_disease` exists
- ✅ OpenAI API key is configured
- ✅ Embeddings are available (or generates them if missing)

### Step 2: Embedding Generation (if needed)

If embeddings are missing (`embedding_built = FALSE`), the system:

1. Loads MONDO diseases and synonyms from `source.mondo_synonym_embeddings`
2. Generates embeddings using OpenAI's `text-embedding-3-small` model
3. Stores embeddings in PostgreSQL vector format
4. Marks records as `embedding_built = TRUE`

### Step 3: Pipeline Initialization

The `LLMDiseaseLinkingPipeline` initializes:

- **OpenAILinker**: Handles disease extraction and embedding-based matching
- **DiseaseValidator**: Rule-based validation filters
- **EnsembleMatcher** (optional): Multi-strategy matching
- **CrossEncoderReranker** (optional): Neural reranking
- **ContextualFilter** (optional): Context-aware filtering

### Step 4: Processing Drug Indications

For each unique product indication:

1. **Query Database**: Groups predicates by `product_name`, selects most recent indication
2. **Disease Extraction**: Extracts disease names from indication text
3. **Matching**: Matches extracted diseases to MONDO IDs
4. **Validation**: Filters invalid matches
5. **Enhancement**: Optional ensemble matching and reranking
6. **Storage**: Saves results to `source.indication_and_disease`

---

## Detailed Example: Processing a Drug Indication

Let's trace through a complete example to understand each step in detail.

### Example Input

**Product Name**: `Truvada`  
**Predicate ID**: `pred_12345`  
**Indication**: `"Treatment of HIV-1 infection in adults and pediatric patients 2 years of age and older who are at high risk for acquiring HIV-1 infection"`

---

### Step-by-Step Processing

#### Step 1: Check Cache

```python
# Check if disease extraction is already cached for this product
cached = _check_extracted_disease("Truvada")
```

**Result**: If cached, returns `"Human Immunodeficiency Virus Infection|HIV-1 infection"` and skips to matching.

**If not cached**, proceeds to extraction.

---

#### Step 2: Disease Extraction with OpenAI

The system sends a prompt to OpenAI GPT-4o-mini:

**System Message**:
```
You are a medical expert specialized in extracting disease names from drug indication text.
Extract diseases that are explicitly mentioned by name in the text, including:
- Primary conditions being treated (e.g., hypertension, diabetes)
- Complications or outcomes being prevented (e.g., stroke, myocardial infarction)
- Comorbid conditions mentioned (e.g., kidney disease, heart failure)
Focus on actual disease names, not symptoms or anatomical parts.
```

**User Prompt**:
```
You are a medical expert. Extract all disease names from the following drug indication text.

Indication: "Treatment of HIV-1 infection in adults and pediatric patients 2 years of age and older who are at high risk for acquiring HIV-1 infection"

Instructions:
1. Extract ALL disease names that are explicitly mentioned in the text...
[... detailed instructions ...]

Disease Names:
```

**API Call** (with prompt caching):
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_completion_tokens=1000,
    prompt_cache_key="disease_extractor_v1",  # Enables prompt caching
)
```

**LLM Response**:
```
Human Immunodeficiency Virus Infection,HIV-1
HIV-1 infection,HIV-1
```

**Parsing**:
- Regex extracts: `"Human Immunodeficiency Virus Infection"` and `"HIV-1 infection"`
- Deduplicates (normalizes to lowercase)
- Stores in cache: `"Human Immunodeficiency Virus Infection|HIV-1 infection"`

**Extracted Diseases**: 
- `["Human Immunodeficiency Virus Infection", "HIV-1 infection"]`

---

#### Step 3: Generate Query Embedding

For each extracted disease, generate an embedding:

```python
query_embedding = _generate_query_embedding("Human Immunodeficiency Virus Infection")
```

**Process**:
- Calls OpenAI Embeddings API with `text-embedding-3-small`
- Returns 1536-dimensional vector
- Converts to PostgreSQL vector format: `[0.123, -0.456, 0.789, ...]`

---

#### Step 4: Vector Similarity Search

Query PostgreSQL using vector similarity (HNSW index):

```sql
SELECT 
    mondo_id,
    disease_name,
    synonym,
    1 - (embedding <=> %s::vector) as similarity
FROM source.mondo_synonym_embeddings
WHERE embedding_built = TRUE
  AND embedding IS NOT NULL
  AND (embedding <=> %s::vector) <= 0.25  -- max_distance = 1 - 0.75
ORDER BY embedding <=> %s::vector
LIMIT 100
```

**Results** (example):
| mondo_id | disease_name | similarity |
|----------|--------------|------------|
| `MONDO:0005091` | `human immunodeficiency virus infectious disease` | 0.92 |
| `MONDO:0005092` | `acquired immunodeficiency syndrome` | 0.85 |
| `MONDO:0005093` | `HIV-1 infection` | 0.88 |

**Top Match**: `MONDO:0005091` with similarity `0.92`

---

#### Step 5: Validation

```python
is_valid, reason = validator.validate_disease(
    "Human Immunodeficiency Virus Infection",
    "MONDO:0005091"
)
```

**Checks**:
- ✅ Not a generic term (e.g., "cancer", "disease NOS")
- ✅ Not obsolete MONDO ID
- ✅ Not a phenotype/process term

**Result**: `(True, "valid")` → Match accepted

---

#### Step 6: Convert to DiseaseMatch

```python
disease_match = DiseaseMatch(
    mondo_id="MONDO:0005091",
    disease_name="human immunodeficiency virus infectious disease",
    matched_phrase="Human Immunodeficiency Virus Infection",
    match_type="openai_embedding",
    confidence=0.92,
    extraction_source="llm",
    similarity=0.92
)
```

---

#### Step 7: Ensemble Matching (Optional)

If enabled, the ensemble matcher combines multiple strategies:

```python
ensemble_results = ensemble_matcher.match(
    query="Human Immunodeficiency Virus Infection",
    semantic_matches=[("MONDO:0005091", 0.92)],
    top_k=3,
    min_confidence=0.7
)
```

**Strategies Applied**:
1. **Exact Match**: Checks if query exactly matches MONDO name/synonym
2. **Fuzzy Match**: RapidFuzz token sort ratio (85% threshold)
3. **Acronym Expansion**: Expands "HIV" → "human immunodeficiency virus"
4. **Semantic Match**: Uses the embedding similarity score

**Result**: If ensemble confirms the match, confidence is boosted:
- `match.confidence = min(1.0, 0.92 * 1.1) = 1.0`
- `match.match_type = "llm_ensemble"`

---

#### Step 8: Cross-Encoder Reranking (Optional)

If enabled, reranks matches using a neural cross-encoder:

```python
reranked = cross_encoder_reranker.rerank(
    query="Treatment of HIV-1 infection in adults...",
    candidates=[
        ("MONDO:0005091", "human immunodeficiency virus infectious disease", 0.92),
        ("MONDO:0005092", "acquired immunodeficiency syndrome", 0.85),
    ],
    top_k=3
)
```

**Process**:
- Encodes query + candidate pairs
- Computes relevance scores
- Reranks by final score
- Filters by `cross_encoder_min_score` threshold

---

#### Step 9: Store Results

Final matches are stored in the database:

```sql
INSERT INTO source.indication_and_disease 
(predicate_id, mondo_disease_ids)
VALUES ('pred_12345', 'MONDO:0005091')
ON CONFLICT (predicate_id) DO UPDATE
SET mondo_disease_ids = EXCLUDED.mondo_disease_ids,
    updated_at = CURRENT_TIMESTAMP
```

**Result**: 
- `predicate_id`: `pred_12345`
- `mondo_disease_ids`: `MONDO:0005091`
- `updated_at`: Current timestamp

---



## Configuration

Key configuration options (via `config.py` or environment variables):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPEN_AI_MODEL` | `gpt-4o-mini` | OpenAI chat model for extraction |
| `OPENAI_EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding model |
| `LLM_TOP_K` | `5` | Max matches per indication |
| `LLM_THRESHOLD` | `0.75` | Minimum similarity score |
| `LLM_MIN_CONFIDENCE` | `0.6` | Minimum ensemble confidence |
| `ENABLE_ENSEMBLE` | `true` | Enable ensemble matching |
| `ENABLE_CONTEXTUAL_FILTER` | `true` | Enable contextual filtering |
| `ENABLE_CROSS_ENCODER_RERANKER` | `true` | Enable cross-encoder reranking |

---

## Key Features

### 1. Prompt Caching
- Uses `prompt_cache_key="disease_extractor_v1"` for all extractions
- Reduces input token costs by caching the system message and prompt structure
- Maximizes cache hit rate across all requests

### 2. Extraction Caching
- Caches extracted diseases per `product_name` in `source.drug_extracted_disease`
- Avoids redundant API calls for the same product
- Speeds up reprocessing

### 3. Vector Similarity Search
- Uses PostgreSQL's `pgvector` extension with HNSW indexing
- Fast similarity search without loading all embeddings into memory
- Cosine similarity threshold filtering

### 4. Incremental Updates
- Groups predicates by `product_name` to process each indication once
- Merges new matches with existing ones
- Updates all associated `predicate_id`s

### 5. Error Handling
- Retry logic for API calls (default: 3 attempts)
- Rate limiting with configurable throttling
- Graceful degradation if optional components fail

---

## Database Schema

### Input Tables

**`drug.drug_predicate_assessments`**
- `id` (predicate_id)
- `product_name`
- `indication` (text to process)

**`source.mondo_tsv_data`**
- `mondo_id`
- `disease_name`
- `synonym`

**`source.mondo_synonym_embeddings`**
- `mondo_id`
- `disease_name`
- `synonym`
- `embedding` (vector)
- `embedding_built` (boolean)

### Output Tables

**`source.indication_and_disease`**
- `predicate_id` (PRIMARY KEY)
- `mondo_disease_ids` (comma-separated)
- `updated_at`

**`source.drug_extracted_disease`** (cache)
- `product_name` (PRIMARY KEY)
- `extracted_disease` (pipe-separated)
- `updated_at`

---

## Running the Pipeline

### Command Line

```bash
# Run LLM join module
python app.py llm_join

# Or directly
cd llm_join
python llm_runner.py
```

### Programmatic Usage

```python
from llm_join.llm_disease_linker import LLMDiseaseLinkingPipeline
import psycopg2

connection = psycopg2.connect(...)

pipeline = LLMDiseaseLinkingPipeline(
    connection=connection,
    openai_api_key="sk-...",
    enable_ensemble=True,
)

# Process single indication
matches = pipeline.link_indication(
    predicate_id="pred_123",
    indication="Treatment of HIV-1 infection",
    product_name="Truvada"
)

# Process all indications
stats = pipeline.run_pipeline()
```

---

## Performance Considerations

1. **API Rate Limiting**: Configured throttling (default: sleep 2s every 10 calls)
2. **Batch Processing**: Processes in batches of 10 products
3. **Vector Index**: HNSW index enables fast similarity search
4. **Caching**: Extraction cache reduces redundant API calls
5. **Prompt Caching**: OpenAI prompt cache reduces input token costs

---

## Troubleshooting

### No Matches Found
- Check embedding generation completed successfully
- Verify `LLM_THRESHOLD` isn't too high
- Check if indication text is valid

### API Errors
- Verify `OPENAI_API_KEY` is set correctly
- Check rate limits and throttling settings
- Review retry logic and error messages

### Missing Embeddings
- Run embedding generator: `OpenAIEmbeddingGenerator.generate_all_embeddings()`
- Check `embedding_built` flag in database

---

## Example Output

For the example indication above, the final result:

```json
{
  "predicate_id": "pred_12345",
  "mondo_disease_ids": "MONDO:0005091",
  "matches": [
    {
      "mondo_id": "MONDO:0005091",
      "disease_name": "human immunodeficiency virus infectious disease",
      "matched_phrase": "Human Immunodeficiency Virus Infection",
      "match_type": "llm_ensemble",
      "confidence": 1.0,
      "similarity": 0.92
    }
  ]
}
```

---

## Summary

The LLM Join pipeline provides a robust, multi-stage approach to linking drug indications to standardized disease ontologies:

1. **Intelligent Extraction**: Uses OpenAI LLM to extract disease names from natural language
2. **Semantic Matching**: Leverages embeddings for accurate disease-to-MONDO matching
3. **Quality Assurance**: Multiple validation and enhancement stages
4. **Efficiency**: Caching and prompt optimization reduce costs and latency
5. **Scalability**: Vector indexing and batch processing handle large datasets

This pipeline transforms unstructured drug indication text into structured, standardized disease identifiers suitable for downstream analysis and integration.

