# Database schema

Auto-generated documentation from schema definitions and live database statistics.

This document covers three databases:
- **Vector store database** (`vector_store.db`): MR-KG embeddings and analysis
- **Trait profile database** (`trait_profile_db.db`): Trait similarity profiles
- **Evidence profile database** (`evidence_profile_db.db`): Evidence similarity profiles

## Database statistics

Live statistics from the actual database files.

### Vector store database

**DuckDB Version:** v1.3.2

| Table/View | Row Count |
|------------|-----------|
| `efo_embeddings` | 67,270 |
| `model_result_traits` | 283,537 |
| `model_results` | 50,402 |
| `mr_pubmed_data` | 15,635 |
| `pmid_model_analysis` | 50,402 |
| `trait_efo_similarity_search` | 5,053,389,670 |
| `trait_embeddings` | 75,121 |
| `trait_similarity_search` | 5,643,089,520 |
| `trait_stats` | 75,121 |

### Trait profile database

**DuckDB Version:** v1.3.2

| Table/View | Row Count |
|------------|-----------|
| `model_similarity_stats` | 6 |
| `query_combinations` | 50,402 |
| `top_similarity_pairs` | 154,115 |
| `trait_similarities` | 504,020 |
| `trait_similarity_analysis` | 504,020 |

### Evidence profile database

**DuckDB Version:** v1.3.2

| Table/View | Row Count |
|------------|-----------|
| `discordant_evidence_pairs` | 1,670 |
| `evidence_similarities` | 6,388 |
| `evidence_similarity_analysis` | 6,388 |
| `high_concordance_pairs` | 4,438 |
| `model_evidence_stats` | 6 |
| `query_combinations` | 6,132 |


## Vector store database

### Overview

```mermaid
erDiagram
    trait_embeddings {
        INTEGER trait_index PK
        VARCHAR trait_label
        FLOAT[200] vector
    }
    efo_embeddings {
        VARCHAR id PK
        VARCHAR label
        FLOAT[200] vector
    }
    model_results {
        INTEGER id PK
        VARCHAR model
        VARCHAR pmid
        JSON metadata
        JSON results
    }
    model_result_traits {
        INTEGER id PK
        INTEGER model_result_id FK
        INTEGER trait_index FK
        VARCHAR trait_label
        VARCHAR trait_id_in_result
    }
    mr_pubmed_data {
        VARCHAR pmid PK
        VARCHAR title
        VARCHAR abstract
        VARCHAR pub_date
        VARCHAR journal
        VARCHAR journal_issn
        VARCHAR author_affil
    }
    trait_similarity_search {
        INTEGER query_id "from_trait_embeddings.trait_index"
        VARCHAR query_label "from_trait_embeddings.trait_label"
        INTEGER result_id "from_trait_embeddings.trait_index"
        VARCHAR result_label "from_trait_embeddings.trait_label"
        FLOAT similarity "from_computed_similarity"
    }
    trait_efo_similarity_search {
        INTEGER trait_index "from_trait_embeddings.trait_index"
        VARCHAR trait_label "from_trait_embeddings.trait_label"
        VARCHAR efo_id "from_efo_embeddings.id"
        VARCHAR efo_label "from_efo_embeddings.label"
        FLOAT similarity "from_computed_similarity"
    }
    pmid_model_analysis {
        VARCHAR pmid "from_model_results.pmid"
        VARCHAR model "from_model_results.model"
        INTEGER model_result_id "from_model_results.id"
        JSON metadata "from_model_results.metadata"
        JSON results "from_model_results.results"
        VARCHAR title "from_mr_pubmed_data.title"
        VARCHAR abstract "from_mr_pubmed_data.abstract"
        VARCHAR pub_date "from_mr_pubmed_data.pub_date"
        VARCHAR journal "from_mr_pubmed_data.journal"
        VARCHAR journal_issn "from_mr_pubmed_data.journal_issn"
        VARCHAR author_affil "from_mr_pubmed_data.author_affil"
        JSON traits "from_aggregated"
    }
    model_result_traits }o--|| model_results : "model_result_id references id"
    model_result_traits }o--|| trait_embeddings : "trait_index references trait_index"
    trait_similarity_search }o..o{ trait_embeddings : "uses"
    trait_efo_similarity_search }o..o{ efo_embeddings : "uses"
    trait_efo_similarity_search }o..o{ trait_embeddings : "uses"
    pmid_model_analysis }o..o{ model_result_traits : "uses"
    pmid_model_analysis }o..o{ model_results : "uses"
    pmid_model_analysis }o..o{ mr_pubmed_data : "uses"

    %% Styling
    style trait_similarity_search fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style trait_efo_similarity_search fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style pmid_model_analysis fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
```

### Quick reference

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `trait_embeddings` | Trait embeddings indexed by unique_traits | trait_index |
| `efo_embeddings` | EFO (Experimental Factor Ontology) term embeddings | id |
| `model_results` | Extracted structural data from model results organized by PMID | id |
| `model_result_traits` | Links model results to traits based on unique_traits indices | id |
| `mr_pubmed_data` | Raw PubMed metadata for papers with MR analysis | pmid |


### Tables

#### trait_embeddings

Trait embeddings indexed by unique_traits.csv indices

**Columns:**

- **`trait_index`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`trait_label`** (VARCHAR, NOT NULL)

- **`vector`** (FLOAT[200], NOT NULL)

#### efo_embeddings

EFO (Experimental Factor Ontology) term embeddings

**Columns:**

- **`id`** (VARCHAR, NOT NULL) (PRIMARY KEY)

- **`label`** (VARCHAR, NOT NULL)

- **`vector`** (FLOAT[200], NOT NULL)

#### model_results

Extracted structural data from model results organized by PMID

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`model`** (VARCHAR, NOT NULL)

- **`pmid`** (VARCHAR, NOT NULL)

- **`metadata`** (JSON, NOT NULL)

- **`results`** (JSON, NOT NULL)

#### model_result_traits

Links model results to traits based on unique_traits indices

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`model_result_id`** (INTEGER, NOT NULL)

- **`trait_index`** (INTEGER, NOT NULL)

- **`trait_label`** (VARCHAR, NOT NULL)

- **`trait_id_in_result`** (VARCHAR, nullable)

**Foreign Keys:**

- `model_result_id` -> `model_results.id`
- `trait_index` -> `trait_embeddings.trait_index`

#### mr_pubmed_data

Raw PubMed metadata for papers with MR analysis

**Columns:**

- **`pmid`** (VARCHAR, NOT NULL) (PRIMARY KEY)

- **`title`** (VARCHAR, NOT NULL)

- **`abstract`** (VARCHAR, NOT NULL)

- **`pub_date`** (VARCHAR, NOT NULL)

- **`journal`** (VARCHAR, NOT NULL)

- **`journal_issn`** (VARCHAR, nullable)

- **`author_affil`** (VARCHAR, nullable)

### Indexes

Performance optimization indexes:

#### efo_embeddings

- **`idx_efo_embeddings_label`** on (label)

#### model_result_traits

- **`idx_model_result_traits_trait_index`** on (trait_index)

- **`idx_model_result_traits_model_result_id`** on (model_result_id)

- **`idx_model_result_traits_trait_label`** on (trait_label)

#### model_results

- **`idx_model_results_model`** on (model)

- **`idx_model_results_pmid`** on (pmid)

#### mr_pubmed_data

- **`idx_mr_pubmed_data_pmid`** on (pmid)

- **`idx_mr_pubmed_data_journal`** on (journal)

- **`idx_mr_pubmed_data_pub_date`** on (pub_date)

#### trait_embeddings

- **`idx_trait_embeddings_label`** on (trait_label)

- **`idx_trait_embeddings_index`** on (trait_index)

### Views

Pre-computed views for common queries:

#### trait_similarity_search

Pre-computed similarity matrix for all trait-to-trait comparisons. Uses cosine similarity on 200-dimensional embeddings to find semantically related traits. Excludes self-comparisons. Useful for discovering related traits and clustering analysis.

**SQL Definition:**

```sql
SELECT
            t1.trait_index as query_id,
            t1.trait_label as query_label,
            t2.trait_index as result_id,
            t2.trait_label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.trait_index != t2.trait_index
```

#### trait_efo_similarity_search

Cross-reference matrix between traits and EFO ontology terms. Uses cosine similarity to map traits to relevant EFO terms for ontology alignment. Enables automatic trait categorization and standardization against biomedical ontologies. Results can be filtered by similarity threshold to find best EFO matches.

**SQL Definition:**

```sql
SELECT
            t.trait_index as trait_index,
            t.trait_label as trait_label,
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
```

#### pmid_model_analysis

Comprehensive view combining PubMed metadata, model results, and extracted traits. Each row is unique per PMID-model combination with traits aggregated into a nested structure. Includes original PubMed data, model metadata/results, and all associated traits. Useful for detailed paper analysis and cross-referencing model outputs with source data.

**SQL Definition:**

```sql
SELECT
            mr.pmid,
            mr.model,
            mr.id as model_result_id,
            mr.metadata,
            mr.results,
            mpd.title,
            mpd.abstract,
            mpd.pub_date,
            mpd.journal,
            mpd.journal_issn,
            mpd.author_affil,
            COALESCE(
                LIST(
                    STRUCT_PACK(
                        trait_index := mrt.trait_index,
                        trait_label := mrt.trait_label,
                        trait_id_in_result := mrt.trait_id_in_result
                    )
                ) FILTER (WHERE mrt.trait_index IS NOT NULL),
                []
            ) as traits
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        GROUP BY 
            mr.pmid, mr.model, mr.id, mr.metadata, mr.results,
            mpd.title, mpd.abstract, mpd.pub_date, mpd.journal, 
            mpd.journal_issn, mpd.author_affil
```


## Trait profile database

### Overview

```mermaid
erDiagram
    query_combinations {
        INTEGER id PK
        VARCHAR pmid
        VARCHAR model
        VARCHAR title
        INTEGER trait_count
    }
    trait_similarities {
        INTEGER id PK
        INTEGER query_combination_id FK
        VARCHAR similar_pmid
        VARCHAR similar_model
        VARCHAR similar_title
        DOUBLE trait_profile_similarity
        DOUBLE trait_jaccard_similarity
        INTEGER query_trait_count
        INTEGER similar_trait_count
    }
    trait_similarity_analysis {
        VARCHAR query_pmid "from_qc.pmid"
        VARCHAR query_model "from_qc.model"
        VARCHAR query_title "from_qc.title"
        VARCHAR query_trait_count "from_qc.trait_count"
        VARCHAR similar_pmid "from_ts.similar_pmid"
        VARCHAR similar_model "from_ts.similar_model"
        VARCHAR similar_title "from_ts.similar_title"
        VARCHAR similar_trait_count "from_ts.similar_trait_count"
        VARCHAR trait_profile_similarity "from_ts.trait_profile_similarity"
        VARCHAR trait_jaccard_similarity "from_ts.trait_jaccard_similarity"
        VARCHAR similarity_rank "from_qc.id"
    }
    model_similarity_stats {
        VARCHAR model "from_unknown"
        VARCHAR total_combinations "from_computed"
        VARCHAR avg_trait_count "from_computed"
        VARCHAR min_trait_count "from_computed"
        VARCHAR max_trait_count "from_computed"
        FLOAT total_similarity_pairs "from_computed"
    }
    top_similarity_pairs {
        VARCHAR model "from_ts.similar_model"
        VARCHAR query_pmid "from_qc.pmid"
        VARCHAR similar_pmid "from_ts.similar_pmid"
        VARCHAR query_title "from_qc.title"
        VARCHAR similar_title "from_ts.similar_title"
        VARCHAR trait_profile_similarity "from_ts.trait_profile_similarity"
        VARCHAR trait_jaccard_similarity "from_ts.trait_jaccard_similarity"
        VARCHAR query_trait_count "from_qc.trait_count"
        VARCHAR similar_trait_count "from_ts.similar_trait_count"
    }
    trait_similarities }o--|| query_combinations : "query_combination_id references id"
    trait_similarity_analysis }o..o{ query_combinations : "uses"
    trait_similarity_analysis }o..o{ trait_similarities : "uses"
    model_similarity_stats }o..o{ query_combinations : "uses"
    top_similarity_pairs }o..o{ query_combinations : "uses"
    top_similarity_pairs }o..o{ trait_similarities : "uses"

    %% Styling
    style trait_similarity_analysis fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style model_similarity_stats fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style top_similarity_pairs fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
```

### Quick reference

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `query_combinations` | PMID-model combinations with trait profile metadata | id |
| `trait_similarities` | Similarity relationships between PMID-model combinations within same model | id |


### Tables

#### query_combinations

PMID-model combinations with trait profile metadata

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`pmid`** (VARCHAR, NOT NULL)

- **`model`** (VARCHAR, NOT NULL)

- **`title`** (VARCHAR, NOT NULL)

- **`trait_count`** (INTEGER, NOT NULL)

#### trait_similarities

Similarity relationships between PMID-model combinations within same model

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`query_combination_id`** (INTEGER, NOT NULL)

- **`similar_pmid`** (VARCHAR, NOT NULL)

- **`similar_model`** (VARCHAR, NOT NULL)

- **`similar_title`** (VARCHAR, NOT NULL)

- **`trait_profile_similarity`** (DOUBLE, NOT NULL)

- **`trait_jaccard_similarity`** (DOUBLE, NOT NULL)

- **`query_trait_count`** (INTEGER, NOT NULL)

- **`similar_trait_count`** (INTEGER, NOT NULL)

**Foreign Keys:**

- `query_combination_id` -> `query_combinations.id`

### Indexes

Performance optimization indexes:

#### query_combinations

- **`idx_query_combinations_pmid`** on (pmid)

- **`idx_query_combinations_model`** on (model)

- **`idx_query_combinations_pmid_model`** on (pmid, model)

#### trait_similarities

- **`idx_trait_similarities_query_id`** on (query_combination_id)

- **`idx_trait_similarities_similar_pmid`** on (similar_pmid)

- **`idx_trait_similarities_similar_model`** on (similar_model)

- **`idx_trait_similarities_trait_profile_sim`** on (trait_profile_similarity)

- **`idx_trait_similarities_jaccard_sim`** on (trait_jaccard_similarity)

### Views

Pre-computed views for common queries:

#### trait_similarity_analysis

**SQL Definition:**

```sql
SELECT
            qc.pmid as query_pmid,
            qc.model as query_model,
            qc.title as query_title,
            qc.trait_count as query_trait_count,
            ts.similar_pmid,
            ts.similar_model,
            ts.similar_title,
            ts.similar_trait_count,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            RANK() OVER (
                PARTITION BY qc.id 
                ORDER BY ts.trait_profile_similarity DESC
            ) as similarity_rank
        FROM query_combinations qc
        JOIN trait_similarities ts ON qc.id = ts.query_combination_id
        ORDER BY qc.pmid, qc.model, ts.trait_profile_similarity DESC
```

#### model_similarity_stats

**SQL Definition:**

```sql
SELECT
            model,
            COUNT(*) as total_combinations,
            AVG(trait_count) as avg_trait_count,
            MIN(trait_count) as min_trait_count,
            MAX(trait_count) as max_trait_count,
            COUNT(*) * 10 as total_similarity_pairs
        FROM query_combinations
        GROUP BY model
        ORDER BY model
```

#### top_similarity_pairs

**SQL Definition:**

```sql
SELECT
            ts.similar_model as model,
            qc.pmid as query_pmid,
            ts.similar_pmid,
            qc.title as query_title,
            ts.similar_title,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            qc.trait_count as query_trait_count,
            ts.similar_trait_count
        FROM trait_similarities ts
        JOIN query_combinations qc ON ts.query_combination_id = qc.id
        WHERE ts.trait_profile_similarity >= 0.8
        ORDER BY ts.similar_model, ts.trait_profile_similarity DESC
```


## Evidence profile database

### Overview

```mermaid
erDiagram
    query_combinations {
        INTEGER id PK
        VARCHAR pmid
        VARCHAR model
        VARCHAR title
        INTEGER result_count
        INTEGER complete_result_count
        DOUBLE data_completeness
        INTEGER publication_year
    }
    evidence_similarities {
        INTEGER id PK
        INTEGER query_combination_id FK
        VARCHAR similar_pmid
        VARCHAR similar_model
        VARCHAR similar_title
        INTEGER matched_pairs
        INTEGER match_type_exact
        INTEGER match_type_fuzzy
        INTEGER match_type_efo
        DOUBLE effect_size_similarity
        DOUBLE direction_concordance
        DOUBLE statistical_consistency
        DOUBLE evidence_overlap
        DOUBLE null_concordance
        DOUBLE effect_size_within_type
        DOUBLE effect_size_cross_type
        INTEGER n_within_type_pairs
        INTEGER n_cross_type_pairs
        INTEGER similar_publication_year
        DOUBLE query_completeness
        DOUBLE similar_completeness
        DOUBLE composite_similarity_equal
        DOUBLE composite_similarity_direction
        INTEGER query_result_count
        INTEGER similar_result_count
    }
    evidence_similarity_analysis {
        VARCHAR query_pmid "from_qc.pmid"
        VARCHAR query_model "from_qc.model"
        VARCHAR query_title "from_qc.title"
        VARCHAR query_result_count "from_qc.result_count"
        VARCHAR query_completeness "from_qc.data_completeness"
        VARCHAR similar_pmid "from_es.similar_pmid"
        VARCHAR similar_model "from_es.similar_model"
        VARCHAR similar_title "from_es.similar_title"
        VARCHAR similar_result_count "from_es.similar_result_count"
        VARCHAR matched_pairs "from_es.matched_pairs"
        VARCHAR match_type_exact "from_es.match_type_exact"
        VARCHAR match_type_fuzzy "from_es.match_type_fuzzy"
        VARCHAR match_type_efo "from_es.match_type_efo"
        VARCHAR effect_size_similarity "from_es.effect_size_similarity"
        VARCHAR direction_concordance "from_es.direction_concordance"
        VARCHAR statistical_consistency "from_es.statistical_consistency"
        VARCHAR evidence_overlap "from_es.evidence_overlap"
        VARCHAR composite_similarity_equal "from_es.composite_similarity_equal"
        VARCHAR composite_similarity_direction "from_es.composite_similarity_direction"
        VARCHAR similarity_rank "from_qc.id"
    }
    model_evidence_stats {
        VARCHAR model "from_unknown"
        VARCHAR total_combinations "from_computed"
        VARCHAR avg_result_count "from_computed"
        VARCHAR avg_completeness "from_computed"
        VARCHAR min_result_count "from_computed"
        VARCHAR max_result_count "from_computed"
        FLOAT total_similarity_pairs "from_computed"
    }
    high_concordance_pairs {
        VARCHAR model "from_es.similar_model"
        VARCHAR query_pmid "from_qc.pmid"
        VARCHAR similar_pmid "from_es.similar_pmid"
        VARCHAR query_title "from_qc.title"
        VARCHAR similar_title "from_es.similar_title"
        VARCHAR direction_concordance "from_es.direction_concordance"
        VARCHAR effect_size_similarity "from_es.effect_size_similarity"
        VARCHAR evidence_overlap "from_es.evidence_overlap"
        VARCHAR matched_pairs "from_es.matched_pairs"
        VARCHAR match_type_exact "from_es.match_type_exact"
        VARCHAR match_type_fuzzy "from_es.match_type_fuzzy"
        VARCHAR match_type_efo "from_es.match_type_efo"
        VARCHAR query_result_count "from_qc.result_count"
        VARCHAR similar_result_count "from_es.similar_result_count"
    }
    discordant_evidence_pairs {
        VARCHAR model "from_es.similar_model"
        VARCHAR query_pmid "from_qc.pmid"
        VARCHAR similar_pmid "from_es.similar_pmid"
        VARCHAR query_title "from_qc.title"
        VARCHAR similar_title "from_es.similar_title"
        VARCHAR direction_concordance "from_es.direction_concordance"
        VARCHAR matched_pairs "from_es.matched_pairs"
        VARCHAR match_type_exact "from_es.match_type_exact"
        VARCHAR match_type_fuzzy "from_es.match_type_fuzzy"
        VARCHAR match_type_efo "from_es.match_type_efo"
        VARCHAR evidence_overlap "from_es.evidence_overlap"
        VARCHAR query_result_count "from_qc.result_count"
        VARCHAR similar_result_count "from_es.similar_result_count"
    }
    match_type_distribution {
        VARCHAR model "from_es.similar_model"
        VARCHAR total_comparisons "from_computed"
        VARCHAR total_exact_matches "from_es.match_type_exact"
        VARCHAR total_fuzzy_matches "from_es.match_type_fuzzy"
        VARCHAR total_efo_matches "from_es.match_type_efo"
        VARCHAR total_matched_pairs "from_es.matched_pairs"
        VARCHAR avg_exact_per_comparison "from_es.match_type_exact"
        VARCHAR avg_fuzzy_per_comparison "from_es.match_type_fuzzy"
        VARCHAR avg_efo_per_comparison "from_es.match_type_efo"
        VARCHAR avg_total_pairs_per_comparison "from_es.matched_pairs"
        VARCHAR pct_exact "from_100.0"
        VARCHAR pct_fuzzy "from_100.0"
        VARCHAR pct_efo "from_100.0"
    }
    evidence_similarities }o--|| query_combinations : "query_combination_id references id"
    evidence_similarity_analysis }o..o{ evidence_similarities : "uses"
    evidence_similarity_analysis }o..o{ query_combinations : "uses"
    model_evidence_stats }o..o{ query_combinations : "uses"
    high_concordance_pairs }o..o{ evidence_similarities : "uses"
    high_concordance_pairs }o..o{ query_combinations : "uses"
    discordant_evidence_pairs }o..o{ evidence_similarities : "uses"
    discordant_evidence_pairs }o..o{ query_combinations : "uses"
    match_type_distribution }o..o{ evidence_similarities : "uses"

    %% Styling
    style evidence_similarity_analysis fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style model_evidence_stats fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style high_concordance_pairs fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style discordant_evidence_pairs fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style match_type_distribution fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
```

### Quick reference

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `query_combinations` | PMID-model combinations with evidence profile metadata and data quality metrics | id |
| `evidence_similarities` | Similarity relationships between PMID-model combinations within same model based on quantitative causal evidence | id |


### Tables

#### query_combinations

PMID-model combinations with evidence profile metadata and data quality metrics

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`pmid`** (VARCHAR, NOT NULL)

- **`model`** (VARCHAR, NOT NULL)

- **`title`** (VARCHAR, NOT NULL)

- **`result_count`** (INTEGER, NOT NULL)

- **`complete_result_count`** (INTEGER, NOT NULL)

- **`data_completeness`** (DOUBLE, NOT NULL)

- **`publication_year`** (INTEGER, nullable)

#### evidence_similarities

Similarity relationships between PMID-model combinations within same model based on quantitative causal evidence

**Columns:**

- **`id`** (INTEGER, NOT NULL) (PRIMARY KEY)

- **`query_combination_id`** (INTEGER, NOT NULL)

- **`similar_pmid`** (VARCHAR, NOT NULL)

- **`similar_model`** (VARCHAR, NOT NULL)

- **`similar_title`** (VARCHAR, NOT NULL)

- **`matched_pairs`** (INTEGER, NOT NULL)

- **`match_type_exact`** (INTEGER, NOT NULL)

- **`match_type_fuzzy`** (INTEGER, NOT NULL)

- **`match_type_efo`** (INTEGER, NOT NULL)

- **`effect_size_similarity`** (DOUBLE, nullable)

- **`direction_concordance`** (DOUBLE, NOT NULL)

- **`statistical_consistency`** (DOUBLE, nullable)

- **`evidence_overlap`** (DOUBLE, NOT NULL)

- **`null_concordance`** (DOUBLE, NOT NULL)

- **`effect_size_within_type`** (DOUBLE, nullable)

- **`effect_size_cross_type`** (DOUBLE, nullable)

- **`n_within_type_pairs`** (INTEGER, NOT NULL)

- **`n_cross_type_pairs`** (INTEGER, NOT NULL)

- **`similar_publication_year`** (INTEGER, nullable)

- **`query_completeness`** (DOUBLE, NOT NULL)

- **`similar_completeness`** (DOUBLE, NOT NULL)

- **`composite_similarity_equal`** (DOUBLE, nullable)

- **`composite_similarity_direction`** (DOUBLE, nullable)

- **`query_result_count`** (INTEGER, NOT NULL)

- **`similar_result_count`** (INTEGER, NOT NULL)

**Foreign Keys:**

- `query_combination_id` -> `query_combinations.id`

### Indexes

Performance optimization indexes:

#### evidence_similarities

- **`idx_evidence_similarities_query_id`** on (query_combination_id)

- **`idx_evidence_similarities_similar_pmid`** on (similar_pmid)

- **`idx_evidence_similarities_similar_model`** on (similar_model)

- **`idx_evidence_similarities_composite_equal`** on (composite_similarity_equal)

- **`idx_evidence_similarities_composite_direction`** on (composite_similarity_direction)

- **`idx_evidence_similarities_direction_concordance`** on (direction_concordance)

- **`idx_evidence_similarities_match_type_exact`** on (match_type_exact)

- **`idx_evidence_similarities_match_type_fuzzy`** on (match_type_fuzzy)

- **`idx_evidence_similarities_match_type_efo`** on (match_type_efo)

#### query_combinations

- **`idx_query_combinations_pmid`** on (pmid)

- **`idx_query_combinations_model`** on (model)

- **`idx_query_combinations_pmid_model`** on (pmid, model)

### Views

Pre-computed views for common queries:

#### evidence_similarity_analysis

**SQL Definition:**

```sql
SELECT
            qc.pmid as query_pmid,
            qc.model as query_model,
            qc.title as query_title,
            qc.result_count as query_result_count,
            qc.data_completeness as query_completeness,
            es.similar_pmid,
            es.similar_model,
            es.similar_title,
            es.similar_result_count,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            es.effect_size_similarity,
            es.direction_concordance,
            es.statistical_consistency,
            es.evidence_overlap,
            es.composite_similarity_equal,
            es.composite_similarity_direction,
            RANK() OVER (
                PARTITION BY qc.id 
                ORDER BY es.composite_similarity_direction DESC
            ) as similarity_rank
        FROM query_combinations qc
        JOIN evidence_similarities es ON qc.id = es.query_combination_id
        ORDER BY qc.pmid, qc.model, es.composite_similarity_direction DESC
```

#### model_evidence_stats

**SQL Definition:**

```sql
SELECT
            model,
            COUNT(*) as total_combinations,
            AVG(result_count) as avg_result_count,
            AVG(data_completeness) as avg_completeness,
            MIN(result_count) as min_result_count,
            MAX(result_count) as max_result_count,
            COUNT(*) * 10 as total_similarity_pairs
        FROM query_combinations
        GROUP BY model
        ORDER BY model
```

#### high_concordance_pairs

**SQL Definition:**

```sql
SELECT
            es.similar_model as model,
            qc.pmid as query_pmid,
            es.similar_pmid,
            qc.title as query_title,
            es.similar_title,
            es.direction_concordance,
            es.effect_size_similarity,
            es.evidence_overlap,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            qc.result_count as query_result_count,
            es.similar_result_count
        FROM evidence_similarities es
        JOIN query_combinations qc ON es.query_combination_id = qc.id
        WHERE es.direction_concordance >= 0.8
        ORDER BY es.similar_model, es.direction_concordance DESC
```

#### discordant_evidence_pairs

**SQL Definition:**

```sql
SELECT
            es.similar_model as model,
            qc.pmid as query_pmid,
            es.similar_pmid,
            qc.title as query_title,
            es.similar_title,
            es.direction_concordance,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            es.evidence_overlap,
            qc.result_count as query_result_count,
            es.similar_result_count
        FROM evidence_similarities es
        JOIN query_combinations qc ON es.query_combination_id = qc.id
        WHERE es.direction_concordance < 0
        ORDER BY es.similar_model, es.direction_concordance ASC
```

#### match_type_distribution

**SQL Definition:**

```sql
SELECT
            es.similar_model as model,
            COUNT(*) as total_comparisons,
            SUM(es.match_type_exact) as total_exact_matches,
            SUM(es.match_type_fuzzy) as total_fuzzy_matches,
            SUM(es.match_type_efo) as total_efo_matches,
            SUM(es.matched_pairs) as total_matched_pairs,
            AVG(es.match_type_exact) as avg_exact_per_comparison,
            AVG(es.match_type_fuzzy) as avg_fuzzy_per_comparison,
            AVG(es.match_type_efo) as avg_efo_per_comparison,
            AVG(es.matched_pairs) as avg_total_pairs_per_comparison,
            ROUND(100.0 * SUM(es.match_type_exact) / NULLIF(SUM(es.matched_pairs), 0), 2) as pct_exact,
            ROUND(100.0 * SUM(es.match_type_fuzzy) / NULLIF(SUM(es.matched_pairs), 0), 2) as pct_fuzzy,
            ROUND(100.0 * SUM(es.match_type_efo) / NULLIF(SUM(es.matched_pairs), 0), 2) as pct_efo
        FROM evidence_similarities es
        GROUP BY es.similar_model
        ORDER BY es.similar_model
```
