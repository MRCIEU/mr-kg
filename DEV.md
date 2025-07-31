# setting up

## models

cd models

wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

extract

## data

wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json

## processing

cd processing

micromamba env create -f environment.yml

TODO: uv?

## frontend

## backend

```
cd backend
uv run uvicorn app.main:app --reload
```
