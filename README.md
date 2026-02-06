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

## Zarr output structure

`pack_to_zarr.py` converts per-protein `.npz` files into one Zarr store per shard.

- **Shard layout (`--shard-level 2`, default):**

  ```text
  <output_root>/
    <shard_1>/
      <shard_2>.zarr/
  ```

- **Shard layout (`--shard-level 1`):**

  ```text
  <output_root>/
    <shard_1>.zarr/
  ```

Each `.zarr` store contains three top-level arrays:

- `emb`: 2D embedding matrix with shape `(total_residues, embed_dim)`. This is the
  concatenation of token-level embeddings for every protein in that shard.
- `offsets`: 1D `int64` array with shape `(num_proteins + 1,)`. For protein `i`,
  its rows in `emb` are `emb[offsets[i]:offsets[i+1]]`.
- `ids`: 1D fixed-width byte string array with shape `(num_proteins,)`.
  `ids[i]` is the protein accession corresponding to the slice defined by `offsets`.

In other words, `ids`, `offsets`, and `emb` are index-aligned so you can recover a
single protein embedding by looking up the accession in `ids` and slicing `emb`
between consecutive `offsets` values.
