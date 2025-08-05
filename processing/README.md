# processing steps for creating mr-kg database

---

# Processing steps
Overarching steps
- main processing
- building the main database
- building the trait profile database

## main processing

Preprocess traits and efo data
```
just preprocess-traits preprocess-efo
```

Perform embeddings
```
just embed-traits embed-efo
```

aggregate embeddings
```
just aggregate-embeddings
```

## building main database

```
just build-main-db
```

## building the trait profile database

compute the trait profile similarities
```
just compute-trait-similarities
```

aggregate trait similarities
```
just aggregate-trait-similarities
```

build trait profile db
```
just build-trait-profile-db
```

---

# `.env` specification

`ACCOUNT_CODE`: HPC account code
