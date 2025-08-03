# data strcuture

raw: raw datasets to be built into a db
raw/llm-results-aggregated: raw llm results as inherited from llm-data-extraction project

db: databases

assets: assets that are to be kept in git
assets/data-schema: jsonschema as inherited from llm-data-extraction project

## important files


### raw

raw/efo/efo.json: efo download from efo github; version 3.80

raw/llm-results-aggregated/*: copied from llm-data-extraction as is

raw/mr-pubmed-data/mr-pubmed-data.json: copied from llm-data-extraction as is

# file tree

```
❯ eza -T
.
├── assets
│   └── data-schema
│       ├── example-data
│       │   ├── metadata.json
│       │   ├── metadata.schema.json
│       │   ├── results.json
│       │   └── results.schema.json
│       └── processed_results
│           ├── metadata.schema.json
│           └── results.schema.json
├── db
│   ├── database-1754063492.db
│   └── restructured-vector-store.db
├── intermediates
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
│   │   │   └── slurm-12432782_9.out
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
│   └── slurm-12432782_9.out
├── processed
│   ├── efo
│   │   └── efo_terms.json
│   ├── embeddings
│   │   ├── efo.json
│   │   └── traits.json
│   ├── model_results
│   │   └── processed_model_results.json
│   └── traits
│       └── unique_traits.csv
├── raw
│   ├── efo
│   │   └── efo.json
│   ├── llm-results-aggregated
│   │   ├── deepseek-r1-distilled
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── gpt-4-1
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── gpt-4o
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── llama3
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── llama3-2
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── logs
│   │   │   ├── deepseek-r1-distilled_schema_validation_errors.log
│   │   │   ├── gpt-4-1_schema_validation_errors.log
│   │   │   ├── gpt-4o_schema_validation_errors.log
│   │   │   ├── llama3-2_schema_validation_errors.log
│   │   │   ├── llama3_schema_validation_errors.log
│   │   │   └── o4-mini_schema_validation_errors.log
│   │   └── o4-mini
│   │       ├── processed_results.json
│   │       ├── processed_results_invalid.json
│   │       ├── processed_results_valid.json
│   │       └── raw_results.json
│   └── mr-pubmed-data
│       └── mr-pubmed-data.json
└── README.md
```
