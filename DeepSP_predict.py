#! /usr/bin/env python3

from argparse import ArgumentParser
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from csv import DictReader
from keras.models import model_from_json

import os
import subprocess
import sys
import pandas as pd
import numpy as np

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
                                             seq=Seq(antibody['Heavy_Chain']),
                                             description=''),
                                   SeqRecord(id=antibody['Name'],
                                             seq=Seq(antibody['Light_Chain']),
                                             description='')
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
        antibodies.append([SeqRecord(id=name, seq=heavy, description=''),
                           SeqRecord(id=name, seq=light, description='')])
    return antibodies


def anarci_align(antibodies: list[tuple[SeqRecord]],
                 output_filename: str = 'seq_aligned_HL.txt') -> str:
    """
    Call ANARCI on both chains of the antibody, then merge the outputs in format
    accepted by downstream analyses. Returns a filename with merged results.
    """
    with open('seq_H.fasta', mode='w') as h_fasta:
        SeqIO.write([x[0] for x in antibodies], h_fasta, 'fasta')
    with open('seq_L.fasta', mode='w') as l_fasta:
        SeqIO.write([x[1] for x in antibodies], l_fasta, 'fasta')
    print(f'Calling ANARCI', file=sys.stderr)
    subprocess.run(['ANARCI', '-i', 'seq_H.fasta', '-o', 'seq_aligned',
                    '-s', 'imgt', '-r', 'heavy', '--csv'])
    subprocess.run(['ANARCI', '-i', 'seq_L.fasta', '-o', 'seq_aligned',
                    '-s', 'imgt', '-r', 'light', '--csv'])
    with open(output_filename, mode="w") as outfile:
        # Process aligned sequences
        # TODO: remove pandas calls, use builtin csv
        infile_H = pd.read_csv('seq_aligned_H.csv')
        infile_L = pd.read_csv('seq_aligned_KL.csv')
        H_inclusion_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                            '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                            '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
                            '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
                            '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
                            '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                            '111', '111A', '111B', '111C', '111D', '111E', '111F', '111G', '111H',
                            '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B', '112A', '112',
                            '113', '114', '115', '116', '117', '118', '119', '120',
                            '121', '122', '123', '124', '125', '126', '127', '128']

        L_inclusion_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                            '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                            '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
                            '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
                            '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
                            '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                            '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                            '121', '122', '123', '124', '125', '126', '127']

        H_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                  '11': 10, '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                  '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29,
                  '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37, '39': 38, '40': 39,
                  '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46, '48': 47, '49': 48, '50': 49,
                  '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55, '57': 56, '58': 57, '59': 58, '60': 59,
                  '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '70': 69,
                  '71': 70, '72': 71, '73': 72, '74': 73, '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79,
                  '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89,
                  '91': 90, '92': 91, '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99,
                  '101': 100, '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107,
                  '109': 108, '110': 109,
                  '111': 110, '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115, '111F': 116, '111G': 117,
                  '111H': 118,
                  '112I': 119, '112H': 120, '112G': 121, '112F': 122, '112E': 123, '112D': 124, '112C': 125, '112B': 126,
                  '112A': 127, '112': 128,
                  '113': 129, '114': 130, '115': 131, '116': 132, '117': 133, '118': 134, '119': 135, '120': 136,
                  '121': 137, '122': 138, '123': 139, '124': 140, '125': 141, '126': 142, '127': 143, '128': 144}

        L_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                  '11': 10, '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                  '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29,
                  '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37, '39': 38, '40': 39,
                  '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46, '48': 47, '49': 48, '50': 49,
                  '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55, '57': 56, '58': 57, '59': 58, '60': 59,
                  '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '70': 69,
                  '71': 70, '72': 71, '73': 72, '74': 73, '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79,
                  '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89,
                  '91': 90, '92': 91, '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99,
                  '101': 100, '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107,
                  '109': 108, '110': 109,
                  '111': 110, '112': 111, '113': 112, '114': 113, '115': 114, '116': 115, '117': 116, '118': 117,
                  '119': 118, '120': 119,
                  '121': 120, '122': 121, '123': 122, '124': 123, '125': 124, '126': 125, '127': 126, '128': 127}

        N_mAbs = len(infile_H["Id"])

        for i in range(N_mAbs):
            H_tmp = 145 * ['-']
            L_tmp = 127 * ['-']
            for col in infile_H.columns:
                if (col in H_inclusion_list):
                    H_tmp[H_dict[col]] = infile_H.iloc[i][col]
            for col in infile_L.columns:
                if (col in L_inclusion_list):
                    L_tmp[L_dict[col]] = infile_L.iloc[i][col]

            aa_string = ''
            for aa in H_tmp + L_tmp:
                aa_string += aa
            outfile.write(infile_H.iloc[i, 0] + " " + aa_string)
            outfile.write("\n")
    # Cleanup
    for filename in ('seq_H.fasta', 'seq_L.fasta',
                 'seq_aligned_H.csv', 'seq_aligned_KL.csv'):
        os.remove(filename)
    # TODO: return prepared data instead of file
    # Also remove seq_aligned_HL.txt
    return output_filename


def load_input_data(filename):
    name_list = []
    seq_list = []
    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
    return name_list, seq_list


def one_hot_encoder(s):
    d = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}

    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1

    return x


def generate_features(name_list, seq_list):
    # sappos
    X = seq_list
    X = [one_hot_encoder(s=x) for x in X]
    X = np.transpose(np.asarray(X), (0, 2, 1))
    X = np.asarray(X)
    json_file = open('Conv1D_regressionSAPpos.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights("Conv1D_regression_SAPpos.h5")
    loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    sap_pos = loaded_model.predict(X)

    # scmpos
    json_file = open('Conv1D_regressionSCMpos.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights("Conv1D_regression_SCMpos.h5")
    loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    scm_pos = loaded_model.predict(X)

    # scmneg
    json_file = open('Conv1D_regressionSCMneg.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights("Conv1D_regression_SCMneg.h5")
    loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    scm_neg = loaded_model.predict(X)

    features = ['Name', 'SAP_pos_CDRH1', 'SAP_pos_CDRH2', 'SAP_pos_CDRH3', 'SAP_pos_CDRL1', 'SAP_pos_CDRL2',
                'SAP_pos_CDRL3', 'SAP_pos_CDR', 'SAP_pos_Hv', 'SAP_pos_Lv', 'SAP_pos_Fv',
                'SCM_pos_CDRH1', 'SCM_pos_CDRH2', 'SCM_pos_CDRH3', 'SCM_pos_CDRL1', 'SCM_pos_CDRL2', 'SCM_pos_CDRL3',
                'SCM_pos_CDR', 'SCM_pos_Hv', 'SCM_pos_Lv', 'SCM_pos_Fv',
                'SCM_neg_CDRH1', 'SCM_neg_CDRH2', 'SCM_neg_CDRH3', 'SCM_neg_CDRL1', 'SCM_neg_CDRL2', 'SCM_neg_CDRL3',
                'SCM_neg_CDR', 'SCM_neg_Hv', 'SCM_neg_Lv', 'SCM_neg_Fv']
    df = pd.concat([pd.DataFrame(name_list), pd.DataFrame(sap_pos), pd.DataFrame(scm_pos), pd.DataFrame(scm_neg)],
                   ignore_index=True, axis=1, );
    df.columns = features
    return df



if __name__ == '__main__':
    parser = ArgumentParser('DeepSP prediction')
    parser.add_argument('-i', type=str, nargs='+',
                        help='Input file(s)')
    parser.add_argument('--in_format', type=str, default='fasta',
                        help='Input format (`fasta` or `csv`)')
    parser.add_argument('-o', type=str, default='out.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    if args.in_format.lower() == 'csv':
        antibodies = process_csv_input(args.i)
    elif args.in_format.lower() == 'fasta':
        antibodies = process_fasta_input(args.i)
    else:
        raise ValueError('Only `csv` and `fasta` (case-insensitive) are valid in_format values')
    print(f'Found {len(antibodies)} antibodies, processing', file=sys.stderr)
    anarci_align(antibodies)
    name_list, seq_list = load_input_data('seq_aligned_HL.txt')
    df = generate_features(name_list, seq_list)
    df.to_csv(args.o, index=False)
