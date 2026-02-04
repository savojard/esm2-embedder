# esm2-embedder

## Preprocess FASTA (collapse identical sequences)

Use `preprocess_fasta.py` to collapse identical sequences into a single entry keyed by a
sequence checksum. It also writes a TSV mapping file to restore the original accession IDs.

```bash
python preprocess_fasta.py input.fasta output.fasta mapping.tsv
```

By default, the checksum uses MD5 (shorter digest). You can pick another hashlib-supported algorithm:

```bash
python preprocess_fasta.py input.fasta output.fasta mapping.tsv --checksum sha1
```
