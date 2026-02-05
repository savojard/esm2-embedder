#!/usr/bin/env python3
"""
Requires packages:
    pip install zarr numcodecs numpy
"""

import argparse
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc


def directory_store(store_path: Path) -> Any:
    """Return a Zarr DirectoryStore compatible with Zarr v2/v3."""
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(str(store_path))
    try:
        from zarr.storage import DirectoryStore
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise ImportError(
            "Zarr DirectoryStore is unavailable; install zarr<3 or upgrade code."
        ) from exc
    return DirectoryStore(str(store_path))


def load_npz_entry(npz_path: Path) -> Tuple[np.ndarray, str]:
    """Load token embeddings and protein id from an NPZ path."""
    for allow_pickle in (False, True):
        try:
            with np.load(npz_path) as npz:
                token_embeddings = npz["token_embeddings"]
                protein_id = str(npz["id"].item())
            return token_embeddings, protein_id
        except ValueError as exc:
            if allow_pickle:
                raise
            logging.warning(
                "Retrying %s with allow_pickle=True due to error: %s",
                npz_path,
                exc,
            )

    raise RuntimeError(f"Failed to load {npz_path}")


def iter_shards(input_root: Path, shard_level: int) -> Iterable[Tuple[str, List[Path]]]:
    """Yield shard labels with their corresponding directories."""
    if shard_level not in (1, 2):
        raise ValueError("shard_level must be 1 or 2.")

    for shard1_path in sorted(p for p in input_root.iterdir() if p.is_dir()):
        shard2_paths = sorted(p for p in shard1_path.iterdir() if p.is_dir())
        if shard_level == 1:
            yield shard1_path.name, shard2_paths
        else:
            for shard2_path in shard2_paths:
                yield f"{shard1_path.name}/{shard2_path.name}", [shard2_path]


def scan_npz_files(shard_paths: Iterable[Path]) -> List[Path]:
    """Return sorted list of .npz files in the provided shard paths."""
    npz_files: List[Path] = []
    for shard_path in shard_paths:
        npz_files.extend(sorted(shard_path.glob("*.npz")))
    return sorted(npz_files)


def pass1_scan(
    npz_files: List[Path],
) -> Tuple[int, int, int, np.dtype, int]:
    """
    First pass: validate shapes/dtypes and compute totals.

    Returns:
        total_residues, num_proteins, embed_dim, dtype, max_id_bytes
    """
    total_residues = 0
    num_proteins = 0
    embed_dim: Optional[int] = None
    dtype: Optional[np.dtype] = None
    max_id_bytes = 0

    for npz_path in npz_files:
        emb, protein_id = load_npz_entry(npz_path)

        if emb.ndim != 2:
            raise ValueError(f"Expected 2D token_embeddings in {npz_path}")

        if embed_dim is None:
            embed_dim = emb.shape[1]
        elif emb.shape[1] != embed_dim:
            raise ValueError(
                f"Inconsistent embedding dim in {npz_path}: "
                f"{emb.shape[1]} != {embed_dim}"
            )

        if dtype is None:
            dtype = emb.dtype
        elif emb.dtype != dtype:
            raise ValueError(
                f"Inconsistent dtype in {npz_path}: {emb.dtype} != {dtype}"
            )

        total_residues += emb.shape[0]
        num_proteins += 1
        max_id_bytes = max(max_id_bytes, len(str(protein_id).encode("utf-8")))

    if embed_dim is None or dtype is None:
        raise ValueError("No embeddings found to scan.")

    return total_residues, num_proteins, embed_dim, dtype, max_id_bytes


def create_store(
    store_path: Path,
    total_residues: int,
    num_proteins: int,
    embed_dim: int,
    dtype: np.dtype,
    max_id_bytes: int,
    chunk_rows: int,
) -> zarr.Group:
    """Create a Zarr store with emb, offsets, and ids arrays."""
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    store = directory_store(store_path)
    root = zarr.group(store=store, overwrite=True)

    root.create_dataset(
        "emb",
        shape=(total_residues, embed_dim),
        chunks=(chunk_rows, embed_dim),
        dtype=dtype,
        compressor=compressor,
    )

    offset_chunk = min(65536, num_proteins + 1)
    root.create_dataset(
        "offsets",
        shape=(num_proteins + 1,),
        chunks=(offset_chunk,),
        dtype=np.int64,
        compressor=compressor,
    )

    ids_chunk = min(65536, num_proteins)
    ids_dtype = np.dtype(f"S{max_id_bytes}")
    root.create_dataset(
        "ids",
        shape=(num_proteins,),
        chunks=(ids_chunk,),
        dtype=ids_dtype,
        compressor=compressor,
    )

    return root


def pass2_write(
    root: zarr.Group,
    npz_files: List[Path],
) -> Tuple[int, int]:
    """Second pass: write embeddings, offsets, and ids into the Zarr store."""
    emb = root["emb"]
    offsets = root["offsets"]
    ids = root["ids"]

    current_offset = 0
    offsets[0] = 0

    for idx, npz_path in enumerate(npz_files):
        token_embeddings, protein_id = load_npz_entry(npz_path)

        length = token_embeddings.shape[0]
        end_offset = current_offset + length

        emb[current_offset:end_offset, :] = token_embeddings
        offsets[idx + 1] = end_offset
        ids[idx] = np.asarray(protein_id.encode("utf-8"), dtype=ids.dtype)

        current_offset = end_offset

    return len(npz_files), current_offset


def process_shard(
    shard_label: str,
    shard_paths: List[Path],
    output_root: Path,
    chunk_rows: int,
    overwrite: bool,
) -> Tuple[str, int, int, bool]:
    """Process a shard directory list and return summary tuple."""
    npz_files = scan_npz_files(shard_paths)
    if not npz_files:
        return (shard_label, 0, 0, False)

    if "/" in shard_label:
        shard1_name, shard2_name = shard_label.split("/", 1)
        store_path = output_root / shard1_name / f"{shard2_name}.zarr"
        store_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        store_path = output_root / f"{shard_label}.zarr"
        store_path.parent.mkdir(parents=True, exist_ok=True)

    if store_path.exists():
        if overwrite:
            shutil.rmtree(store_path)
        else:
            logging.info("Skipping existing store %s", store_path)
            return (str(store_path.relative_to(output_root)), 0, 0, False)

    total_residues, num_proteins, embed_dim, dtype, max_id_bytes = pass1_scan(
        npz_files
    )
    root = create_store(
        store_path,
        total_residues,
        num_proteins,
        embed_dim,
        dtype,
        max_id_bytes,
        chunk_rows,
    )

    proteins_written, residues_written = pass2_write(root, npz_files)
    return (str(store_path.relative_to(output_root)), proteins_written, residues_written, True)


def sample_npz_paths(input_root: Path, sample_size: int, shard_level: int) -> List[Path]:
    """Reservoir sample npz files without loading the full list into memory."""
    rng = random.Random(42)
    reservoir: List[Path] = []
    seen = 0

    for _, shard_paths in iter_shards(input_root, shard_level):
        for npz_path in scan_npz_files(shard_paths):
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(npz_path)
            else:
                idx = rng.randint(0, seen - 1)
                if idx < sample_size:
                    reservoir[idx] = npz_path

    return reservoir


def load_protein(output_root: str, accession: str) -> np.ndarray:
    """Load a protein embedding from Zarr stores by accession."""
    output_path = Path(output_root)
    accession_str = str(accession)

    store_paths = sorted(output_path.glob("*.zarr")) + sorted(output_path.glob("*/*.zarr"))
    for store_path in store_paths:
        root = zarr.open_group(store_path, mode="r")
        ids = root["ids"][:]
        if ids.dtype.kind == "S":
            target = accession_str.encode("utf-8")
        else:
            target = accession_str
        matches = np.where(ids == target)[0]
        if matches.size > 0:
            idx = int(matches[0])
            offsets = root["offsets"][:]
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            return root["emb"][start:end, :]

    raise KeyError(f"Accession {accession_str} not found in any store.")


def run_check(
    input_root: Path, output_root: Path, sample_size: int, shard_level: int
) -> None:
    """Validate that sampled proteins round-trip correctly."""
    sample_paths = sample_npz_paths(input_root, sample_size, shard_level)
    if not sample_paths:
        logging.warning("No npz files found for validation.")
        return

    failures = 0
    for npz_path in sample_paths:
        token_embeddings, protein_id = load_npz_entry(npz_path)

        retrieved = load_protein(str(output_root), protein_id)

        if retrieved.shape != token_embeddings.shape or not np.array_equal(
            retrieved, token_embeddings
        ):
            failures += 1
            logging.error("Validation failed for %s", protein_id)

    passed = len(sample_paths) - failures
    logging.info("Validation: %d passed, %d failed", passed, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack per-protein NPZ embeddings into sharded Zarr stores."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input root directory")
    parser.add_argument("--output", required=True, type=Path, help="Output root directory")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help="Chunk rows for embedding array",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr stores",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--shard-level",
        type=int,
        default=2,
        choices=(1, 2),
        help="Shard level for output Zarr stores (1 or 2)",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        help="Randomly sample K proteins to validate round-trip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_root = args.input
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    shard_groups = list(iter_shards(input_root, args.shard_level))
    if not shard_groups:
        logging.warning("No shard directories found under %s", input_root)
        return

    total_proteins = 0
    total_residues = 0
    processed_shards = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_shard,
                shard_label,
                shard_paths,
                output_root,
                args.chunk_rows,
                args.overwrite,
            ): shard_label
            for shard_label, shard_paths in shard_groups
        }

        for future in as_completed(futures):
            shard_label = futures[future]
            try:
                store_name, proteins_written, residues_written, created = future.result()
            except Exception as exc:
                logging.error("Shard %s failed: %s", shard_label, exc)
                continue

            if created and proteins_written > 0:
                processed_shards += 1
                total_proteins += proteins_written
                total_residues += residues_written
                logging.info(
                    "Packed shard %s -> %s (%d proteins, %d residues)",
                    shard_label,
                    store_name,
                    proteins_written,
                    residues_written,
                )

    logging.info(
        "Finished: %d shards, %d proteins, %d residues",
        processed_shards,
        total_proteins,
        total_residues,
    )

    if args.check > 0:
        run_check(input_root, output_root, args.check, args.shard_level)


if __name__ == "__main__":
    main()
