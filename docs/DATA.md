Note: This document is the canonical reference for the data directory layout
and database schema artifacts. Processing pipeline steps are documented in
@processing/README.md.

# data strcuture

Top-level layout

- raw/: source datasets ingested as-is

    - efo/
        - efo.json — EFO ontology JSON (v3.80; from upstream EFO release)

    - llm-results-aggregated/ — aggregated LLM outputs from sibling repo (llm-data-extraction)
        - {model}/
            - raw_results.json — original model responses
            - processed_results.json — schema-validated and normalized results
            - processed_results_valid.json — passed schema validation
            - processed_results_invalid.json — failed schema validation
        - logs/
            - {model}_schema_validation_errors.log — JSON Schema validation errors per model

    - mr-pubmed-data/
        - mr-pubmed-data.json — PubMed MR corpus (copied)

- processed/: derived artifacts used to build the databases and drive the app

    - traits/
        - unique_traits.csv — deduplicated trait labels extracted from model outputs

    - embeddings/
        - traits.json — embeddings for trait labels
        - efo.json — embeddings for EFO terms

    - model_results/
        - processed_model_results.json — consolidated, cleaned model outputs

    - trait-profile-similarities/
        - trait-profile-similarities.json — pairwise similarity results
        - aggregation_stats.json — summary statistics over similarities

    - efo/
        - efo_terms.json — parsed EFO terms (for lookup and display)

- db/: DuckDB databases built from processed artifacts

    - vector_store.db — vectors, nearest-neighbor indexes, and model_result_traits
    - trait_profile_db.db — trait profile aggregates and similarity/search views
    - backup-*.db — dated backups of the above

- assets/: versioned schemas and reference files stored in Git

    - data-schema/
        - example-data/
            - metadata.json
            - metadata.schema.json
            - results.json
            - results.schema.json
        - processed_results/
            - metadata.schema.json
            - results.schema.json
    - database_schema/
        - database_info.txt — notes/specs about DB structure

- output/: batch logs and chunked intermediate results from cluster jobs (audit/debug only)

Inspecting databases

- Use webapp/scripts/describe-db.py (just describe-db) to list tables, views, schemas, and row counts.

# file tree (2025-08-10)

```
❯ eza -T ./data
./data
├── assets
│   ├── data-schema
│   │   ├── example-data
│   │   │   ├── metadata.json
│   │   │   ├── metadata.schema.json
│   │   │   ├── results.json
│   │   │   └── results.schema.json
│   │   └── processed_results
│   │       ├── metadata.schema.json
│   │       └── results.schema.json
│   └── database_schema
│       └── database_info.txt
├── db
│   ├── backup-20250809-trait_profile_db.db
│   ├── backup-20250809-vector_store.db
│   ├── trait_profile_db.db
│   └── vector_store.db
├── output
│   ├── bc4-12431897
│   │   ├── logs
│   │   │   ├── script-12431897.out
│   │   │   ├── slurm-12431897_0.out
│   │   │   ├── slurm-12431897_1.out
│   │   │   ├── slurm-12431897_2.out
│   │   │   ├── slurm-12431897_3.out
│   │   │   └── slurm-12431897_4.out
│   │   ├── README
│   │   └── results
│   │       ├── trait_vectors_chunk_0.json
│   │       ├── trait_vectors_chunk_2.json
│   │       ├── trait_vectors_chunk_3.json
│   │       └── trait_vectors_chunk_4.json
│   ├── bc4-12432782
│   │   ├── logs
│   │   │   ├── script-12432782.out
│   │   │   ├── slurm-12432782_0.out
│   │   │   ├── slurm-12432782_1.out
│   │   │   ├── slurm-12432782_2.out
│   │   │   ├── slurm-12432782_3.out
│   │   │   ├── slurm-12432782_4.out
│   │   │   ├── slurm-12432782_5.out
│   │   │   ├── slurm-12432782_6.out
│   │   │   ├── slurm-12432782_7.out
│   │   │   ├── slurm-12432782_8.out
│   │   │   ├── slurm-12432782_9.out
│   │   ├── README
│   │   └── results
│   │       ├── efo_vectors_chunk_0.json
│   │       ├── efo_vectors_chunk_1.json
│   │       ├── efo_vectors_chunk_2.json
│   │       ├── efo_vectors_chunk_3.json
│   │       ├── efo_vectors_chunk_4.json
│   │       ├── efo_vectors_chunk_5.json
│   │       ├── efo_vectors_chunk_6.json
│   │       ├── efo_vectors_chunk_7.json
│   │       ├── efo_vectors_chunk_8.json
│   │       └── efo_vectors_chunk_9.json
│   ├── bc4-12440480
│   │   ├── logs
│   │   │   ├── script-12440480.out
│   │   │   ├── slurm-12440480_0.out
│   │   │   ├── slurm-12440480_1.out
│   │   │   ├── slurm-12440480_2.out
│   │   │   ├── slurm-12440480_3.out
│   │   │   ├── slurm-12440480_4.out
│   │   │   ├── slurm-12440480_5.out
│   │   │   ├── slurm-12440480_6.out
│   │   │   ├── slurm-12440480_7.out
│   │   │   ├── slurm-12440480_8.out
│   │   │   ├── slurm-12440480_9.out
│   │   │   ├── slurm-12440480_10.out
│   │   │   ├── slurm-12440480_11.out
│   │   │   ├── slurm-12440480_12.out
│   │   │   ├── slurm-12440480_13.out
│   │   │   ├── slurm-12440480_14.out
│   │   │   ├── slurm-12440480_15.out
│   │   │   ├── slurm-12440480_16.out
│   │   │   ├── slurm-12440480_17.out
│   │   │   ├── slurm-12440480_18.out
│   │   │   └── slurm-12440480_19.out
│   │   └── results
│   ├── bc4-12440505
│   │   ├── logs
│   │   │   ├── script-12440505.out
│   │   │   ├── slurm-12440505_0.out
│   │   │   ├── slurm-12440505_1.out
│   │   │   ├── slurm-12440505_2.out
│   │   │   ├── slurm-12440505_3.out
│   │   │   ├── slurm-12440505_4.out
│   │   │   ├── slurm-12440505_5.out
│   │   │   ├── slurm-12440505_6.out
│   │   │   ├── slurm-12440505_7.out
│   │   │   ├── slurm-12440505_8.out
│   │   │   ├── slurm-12440505_9.out
│   │   │   ├── slurm-12440505_10.out
│   │   │   ├── slurm-12440505_11.out
│   │   │   ├── slurm-12440505_12.out
│   │   │   ├── slurm-12440505_13.out
│   │   │   ├── slurm-12440505_14.out
│   │   │   ├── slurm-12440505_15.out
│   │   │   ├── slurm-12440505_16.out
│   │   │   ├── slurm-12440505_17.out
│   │   │   ├── slurm-12440505_18.out
│   │   │   └── slurm-12440505_19.out
│   │   ├── README
│   │   └── results
│   │       ├── trait_similarities_chunk_0.json
│   │       ├── trait_similarities_chunk_1.json
│   │       ├── trait_similarities_chunk_2.json
│   │       ├── trait_similarities_chunk_3.json
│   │       ├── trait_similarities_chunk_4.json
│   │       ├── trait_similarities_chunk_5.json
│   │       ├── trait_similarities_chunk_6.json
│   │       ├── trait_similarities_chunk_7.json
│   │       ├── trait_similarities_chunk_8.json
│   │       ├── trait_similarities_chunk_9.json
│   │       ├── trait_similarities_chunk_10.json
│   │       ├── trait_similarities_chunk_11.json
│   │       ├── trait_similarities_chunk_12.json
│   │       ├── trait_similarities_chunk_13.json
│   │       ├── trait_similarities_chunk_14.json
│   │       ├── trait_similarities_chunk_15.json
│   │       ├── trait_similarities_chunk_16.json
│   │       ├── trait_similarities_chunk_17.json
│   │       └── trait_similarities_chunk_19.json
│   ├── bc4-12467639
│   │   ├── logs
│   │   │   ├── script-12467639.out
│   │   │   ├── slurm-12467639_0.out
│   │   │   ├── slurm-12467639_1.out
│   │   │   ├── slurm-12467639_2.out
│   │   │   ├── slurm-12467639_3.out
│   │   │   └── slurm-12467639_4.out
│   │   ├── README
│   │   └── results
│   │       ├── trait_vectors_chunk_0.json
│   │       ├── trait_vectors_chunk_2.json
│   │       ├── trait_vectors_chunk_3.json
│   │       └── trait_vectors_chunk_4.json
│   ├── bc4-12468629
│   │   ├── logs
│   │   │   ├── script-12468629.out
│   │   │   ├── slurm-12468629_0.out
│   │   │   ├── slurm-12468629_1.out
│   │   │   ├── slurm-12468629_2.out
│   │   │   ├── slurm-12468629_3.out
│   │   │   ├── slurm-12468629_4.out
│   │   │   ├── slurm-12468629_5.out
│   │   │   ├── slurm-12468629_6.out
│   │   │   ├── slurm-12468629_7.out
│   │   │   ├── slurm-12468629_8.out
│   │   │   ├── slurm-12468629_9.out
│   │   │   ├── slurm-12468629_10.out
│   │   │   ├── slurm-12468629_11.out
│   │   │   ├── slurm-12468629_12.out
│   │   │   ├── slurm-12468629_13.out
│   │   │   ├── slurm-12468629_14.out
│   │   │   ├── slurm-12468629_15.out
│   │   │   ├── slurm-12468629_16.out
│   │   │   ├── slurm-12468629_17.out
│   │   │   ├── slurm-12468629_18.out
│   │   │   └── slurm-12468629_19.out
│   │   ├── README
│   │   └── results
│   │       ├── trait_similarities_chunk_0.json
│   │       ├── trait_similarities_chunk_1.json
│   │       ├── trait_similarities_chunk_2.json
│   │       ├── trait_similarities_chunk_3.json
│   │       ├── trait_similarities_chunk_4.json
│   │       ├── trait_similarities_chunk_5.json
│   │       ├── trait_similarities_chunk_6.json
│   │       ├── trait_similarities_chunk_7.json
│   │       ├── trait_similarities_chunk_8.json
│   │       ├── trait_similarities_chunk_9.json
│   │       ├── trait_similarities_chunk_10.json
│   │       ├── trait_similarities_chunk_11.json
│   │       ├── trait_similarities_chunk_12.json
│   │       ├── trait_similarities_chunk_13.json
│   │       ├── trait_similarities_chunk_14.json
│   │       ├── trait_similarities_chunk_15.json
│   │       ├── trait_similarities_chunk_16.json
│   │       ├── trait_similarities_chunk_17.json
│   │       ├── trait_similarities_chunk_18.json
│   │       └── trait_similarities_chunk_19.json
│   ├── slurm-12431897_0.out
│   ├── slurm-12431897_1.out
│   ├── slurm-12431897_2.out
│   ├── slurm-12431897_3.out
│   ├── slurm-12431897_4.out
│   ├── slurm-12432782_0.out
│   ├── slurm-12432782_1.out
│   ├── slurm-12432782_2.out
│   ├── slurm-12432782_3.out
│   ├── slurm-12432782_4.out
│   ├── slurm-12432782_5.out
│   ├── slurm-12432782_6.out
│   ├── slurm-12432782_7.out
│   ├── slurm-12432782_8.out
│   ├── slurm-12432782_9.out
│   ├── slurm-12440480_0.out
│   ├── slurm-12440480_1.out
│   ├── slurm-12440480_2.out
│   ├── slurm-12440480_3.out
│   ├── slurm-12440480_4.out
│   ├── slurm-12440480_5.out
│   ├── slurm-12440480_6.out
│   ├── slurm-12440480_7.out
│   ├── slurm-12440480_8.out
│   ├── slurm-12440480_9.out
│   ├── slurm-12440480_10.out
│   ├── slurm-12440480_11.out
│   ├── slurm-12440480_12.out
│   ├── slurm-12440480_13.out
│   ├── slurm-12440480_14.out
│   ├── slurm-12440480_15.out
│   ├── slurm-12440480_16.out
│   ├── slurm-12440480_17.out
│   ├── slurm-12440480_18.out
│   ├── slurm-12440480_19.out
│   ├── slurm-12440505_0.out
│   ├── slurm-12440505_1.out
│   ├── slurm-12440505_2.out
│   ├── slurm-12440505_3.out
│   ├── slurm-12440505_4.out
│   ├── slurm-12440505_5.out
│   ├── slurm-12440505_6.out
│   ├── slurm-12440505_7.out
│   ├── slurm-12440505_8.out
│   ├── slurm-12440505_9.out
│   ├── slurm-12440505_10.out
│   ├── slurm-12440505_11.out
│   ├── slurm-12440505_12.out
│   ├── slurm-12440505_13.out
│   ├── slurm-12440505_14.out
│   ├── slurm-12440505_15.out
│   ├── slurm-12440505_16.out
│   ├── slurm-12440505_17.out
│   ├── slurm-12440505_18.out
│   ├── slurm-12440505_19.out
│   ├── slurm-12467639_0.out
│   ├── slurm-12467639_1.out
│   ├── slurm-12467639_2.out
│   ├── slurm-12467639_3.out
│   ├── slurm-12467639_4.out
│   ├── slurm-12468629_0.out
│   ├── slurm-12468629_1.out
│   ├── slurm-12468629_2.out
│   ├── slurm-12468629_3.out
│   ├── slurm-12468629_4.out
│   ├── slurm-12468629_5.out
│   ├── slurm-12468629_6.out
│   ├── slurm-12468629_7.out
│   ├── slurm-12468629_8.out
│   ├── slurm-12468629_9.out
│   ├── slurm-12468629_10.out
│   ├── slurm-12468629_11.out
│   ├── slurm-12468629_12.out
│   ├── slurm-12468629_13.out
│   ├── slurm-12468629_14.out
│   ├── slurm-12468629_15.out
│   ├── slurm-12468629_16.out
│   ├── slurm-12468629_17.out
│   ├── slurm-12468629_18.out
│   └── slurm-12468629_19.out
├── processed
│   ├── efo
│   │   └── efo_terms.json
│   ├── embeddings
│   │   ├── efo.json
│   │   └── traits.json
│   ├── model_results
│   │   └── processed_model_results.json
│   ├── trait-profile-similarities
│   │   ├── aggregation_stats.json
│   │   └── trait-profile-similarities.json
│   └── traits
│       └── unique_traits.csv
└── raw
    ├── efo
    │   └── efo.json
    ├── llm-results-aggregated
    │   ├── deepseek-r1-distilled
    │   │   ├── processed_results.json
    │   │   ├── processed_results_invalid.json
    │   │   ├── processed_results_valid.json
    │   │   └── raw_results.json
    │   ├── gpt-4-1
    │   │   ├── processed_results.json
    │   │   ├── processed_results_invalid.json
    │   │   ├── processed_results_valid.json
    │   │   └── raw_results.json
    │   ├── gpt-4o
    │   │   ├── processed_results.json
    │   │   ├── processed_results_invalid.json
    │   │   ├── processed_results_valid.json
    │   │   └── raw_results.json
    │   ├── llama3
    │   │   ├── processed_results.json
    │   │   ├── processed_results_invalid.json
    │   │   ├── processed_results_valid.json
    │   │   └── raw_results.json
    │   ├── llama3-2
    │   │   ├── processed_results.json
    │   │   ├── processed_results_invalid.json
    │   │   ├── processed_results_valid.json
    │   │   └── raw_results.json
    │   ├── logs
    │   │   ├── deepseek-r1-distilled_schema_validation_errors.log
    │   │   ├── gpt-4-1_schema_validation_errors.log
    │   │   ├── gpt-4o_schema_validation_errors.log
    │   │   ├── llama3-2_schema_validation_errors.log
    │   │   ├── llama3_schema_validation_errors.log
    │   │   └── o4-mini_schema_validation_errors.log
    │   └── o4-mini
    │       ├── processed_results.json
    │       ├── processed_results_invalid.json
    │       ├── processed_results_valid.json
    │       └── raw_results.json
    └── mr-pubmed-data
        └── mr-pubmed-data.json
```
