CREATE TABLE IF NOT EXISTS source.mondo_tsv_data (
    mondo_id VARCHAR(255) PRIMARY KEY,
    disease_name VARCHAR(255),
    description TEXT,
    synonym TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mondo_id ON source.mondo_tsv_data(mondo_id);
CREATE INDEX IF NOT EXISTS idx_disease_name ON source.mondo_tsv_data(disease_name);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'uq_mondo_mondo_id'
    ) THEN
        ALTER TABLE source.mondo_tsv_data
        ADD CONSTRAINT uq_mondo_mondo_id
            UNIQUE NULLS NOT DISTINCT (mondo_id);
    END IF;
END $$;
    

CREATE TABLE IF NOT EXISTS source.indication_and_disease (
    id SERIAL PRIMARY KEY,
    predicate_id INT UNIQUE REFERENCES drug.drug_predicate_assessments(id),
    mondo_disease_ids TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predicate_id ON source.indication_and_disease(predicate_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'uq_indication_and_disease'
    ) THEN
        ALTER TABLE source.indication_and_disease
        ADD CONSTRAINT uq_indication_and_disease
            UNIQUE NULLS NOT DISTINCT (
                predicate_id
            );
    END IF;
END $$;


CREATE TABLE IF NOT EXISTS source.drug_extracted_disease (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(500) UNIQUE,
    extracted_disease TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_product_name ON source.drug_extracted_disease (product_name);




CREATE TABLE IF NOT EXISTS source.mondo_synonym_embeddings ( 
    mondo_id TEXT, 
    disease_name TEXT,
    synonym TEXT, 
    embedding vector(1536), 
    embedding_built BOOLEAN DEFAULT FALSE, 
    PRIMARY KEY (mondo_id, synonym) 
    );
CREATE INDEX IF NOT EXISTS idx_mondo_embedding_hnsw ON source.mondo_synonym_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_mondo_synonym_mondo_id ON source.mondo_synonym_embeddings (mondo_id);
CREATE INDEX IF NOT EXISTS idx_embedding_built ON source.mondo_synonym_embeddings (embedding_built);

