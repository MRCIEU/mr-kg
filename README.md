# MR-KG: Mendelian Randomization Knowledge Graph

This repository implements MR-KG (Mendelian Randomization Knowledge Graph), a
system for processing and exploring Mendelian Randomization studies through
large language model-extracted trait information and vector similarity search.

## Components

- **Processing pipeline**: ETL pipeline that creates DuckDB databases from raw
  LLM results and EFO ontology data
- **Webapp**: Streamlit interface for exploring the processed data

## Quick start

```bash
git clone https://github.com/MRCIEU/mr-kg
cd mr-kg
just setup-dev
just dev
```

Access the webapp at http://localhost:8501

## Documentation

- Development guide: @DEV.md
- Data structure: @docs/DATA.md
- Key terms and concepts: @docs/GLOSSARY.md
- Processing pipeline: @docs/processing/pipeline.md
- Case study analyses: @docs/processing/case-studies.md

## Citation

TBC
