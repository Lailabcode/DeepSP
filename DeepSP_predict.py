#! /usr/bin/env python3

from argparse import ArgumentParser
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from csv import DictReader


def process_csv_input(files: list[str]) -> list[tuple[SeqRecord]]:
    """
    Take antibody sequences from a CSV input.

    Assumes that column names are 'Name,Heavy_Chain,Light_Chain'; other columns,
    if any, are ignored.
    """
    antibodies = []
    for csv_file in files:
        with open(csv_file) as csv_input:
            reader = DictReader(csv_input)
            for antibody in reader:
                antibodies.append([SeqRecord(id=antibody['Name'],
                                            seq=Seq(antibody['Heavy_Chain'])),
                                   SeqRecord(id=antibody['Name'],
                                             seq=Seq(antibody['Light_Chain']))
                                   ])
    return antibodies


def process_fasta_input(files: list[str]) -> list[tuple[SeqRecord]]:
    """
    Take antibody sequences from FASTA input.

    Assumes that each FASTA file contains two records named '{ab_id}_VH' and
    '{ab_id}_VL'. Raises exception if any of these records is absent or any
    additional records are found; order of records in the file does not matter.
    """
    antibodies = []
    for fasta_file in args.i:
        name = ''
        heavy = ''
        light = ''
        for record in SeqIO.parse(fasta_file, 'fasta'):
            if record.id.endswith('_VH'):
                name = '_'.join(record.id.split('_')[:-1])
                heavy = record.seq
            elif record.id.endswith('_VL'):
                light = record.seq
            else:
                raise ValueError(f'Invalid postfix in FASTA record name {record.id}')
        antibodies.append([SeqRecord(id=name, seq=heavy),
                           SeqRecord(id=name, seq=light)])
    return antibodies


if __name__ == '__main__':
    parser = ArgumentParser('DeepSP prediction')
    parser.add_argument('-i', type=str, nargs='+',
                        help='Input file(s)')
    parser.add_argument('--in_format', type=str, default='fasta',
                        help='Input format (`fasta` or `csv`)')
    parser.add_argument('-o', type=str, help='Output CSV path')
    args = parser.parse_args()
    if args.in_format.lower() == 'csv':
        antibodies = process_csv_input(args.i)
    elif args.in_format.lower() == 'fasta':
        antibodies = process_fasta_input(args.i)
    else:
        raise ValueError('Only `csv` and `fasta` (case-insensitive) are valid in_format values')
    print(antibodies)
    print(len(antibodies))