#!/usr/bin/env python
"""
Collapse identical FASTA sequences into a checksum-based FASTA.

Outputs:
- FASTA with one entry per unique sequence (id = checksum)
- TSV mapping of original accessions to checksum
"""
import argparse
import hashlib
from typing import Dict, Iterable, List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def normalize_sequence(raw: str) -> str:
    return raw.replace(" ", "").replace("\n", "").upper()


def sequence_checksum(sequence: str, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(sequence.encode("utf-8"))
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse identical FASTA sequences into checksum-identified entries."
    )
    parser.add_argument("input_fasta", help="Input FASTA file.")
    parser.add_argument("output_fasta", help="Output FASTA with unique sequences.")
    parser.add_argument("mapping_tsv", help="Output TSV mapping accession to checksum.")
    parser.add_argument(
        "--checksum",
        default="md5",
        help="Checksum algorithm for sequence IDs (default: md5).",
    )
    return parser.parse_args()


def collapse_fasta(
    fasta_path: str, algorithm: str
) -> Tuple[List[SeqRecord], List[Tuple[str, str]]]:
    unique_sequences: Dict[str, str] = {}
    mapping: List[Tuple[str, str]] = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence = normalize_sequence(str(record.seq))
        if not sequence:
            continue
        checksum = sequence_checksum(sequence, algorithm)
        mapping.append((record.id, checksum))
        if checksum not in unique_sequences:
            unique_sequences[checksum] = sequence

    if not unique_sequences:
        raise ValueError("No sequences found in FASTA.")

    unique_records = [
        SeqRecord(Seq(seq), id=checksum, description="")
        for checksum, seq in unique_sequences.items()
    ]
    return unique_records, mapping


def write_mapping(mapping_tsv: str, mapping: Iterable[Tuple[str, str]]) -> None:
    with open(mapping_tsv, "w", encoding="utf-8") as handle:
        handle.write("accession\tchecksum\n")
        for accession, checksum in mapping:
            handle.write(f"{accession}\t{checksum}\n")


def main() -> None:
    args = parse_args()
    records, mapping = collapse_fasta(args.input_fasta, args.checksum)
    SeqIO.write(records, args.output_fasta, "fasta")
    write_mapping(args.mapping_tsv, mapping)


if __name__ == "__main__":
    main()
