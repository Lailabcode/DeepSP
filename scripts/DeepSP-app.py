# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:32:38 2023

@author: plai3
"""

import streamlit as st
import numpy as np
import pandas as pd

from keras.models import model_from_json

from Bio import SeqIO
from io import StringIO
from anarci import anarci


def one_hot_encoder(s):
    d = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "-": 20,
    }

    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1

    return x


H_inclusion_list = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "63",
    "64",
    "65",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "91",
    "92",
    "93",
    "94",
    "95",
    "96",
    "97",
    "98",
    "99",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "111A",
    "111B",
    "111C",
    "111D",
    "111E",
    "111F",
    "111G",
    "111H",
    "112I",
    "112H",
    "112G",
    "112F",
    "112E",
    "112D",
    "112C",
    "112B",
    "112A",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
    "123",
    "124",
    "125",
    "126",
    "127",
    "128",
]

L_inclusion_list = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "63",
    "64",
    "65",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "91",
    "92",
    "93",
    "94",
    "95",
    "96",
    "97",
    "98",
    "99",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
    "123",
    "124",
    "125",
    "126",
    "127",
]

H_dict = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "11": 10,
    "12": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "19": 18,
    "20": 19,
    "21": 20,
    "22": 21,
    "23": 22,
    "24": 23,
    "25": 24,
    "26": 25,
    "27": 26,
    "28": 27,
    "29": 28,
    "30": 29,
    "31": 30,
    "32": 31,
    "33": 32,
    "34": 33,
    "35": 34,
    "36": 35,
    "37": 36,
    "38": 37,
    "39": 38,
    "40": 39,
    "41": 40,
    "42": 41,
    "43": 42,
    "44": 43,
    "45": 44,
    "46": 45,
    "47": 46,
    "48": 47,
    "49": 48,
    "50": 49,
    "51": 50,
    "52": 51,
    "53": 52,
    "54": 53,
    "55": 54,
    "56": 55,
    "57": 56,
    "58": 57,
    "59": 58,
    "60": 59,
    "61": 60,
    "62": 61,
    "63": 62,
    "64": 63,
    "65": 64,
    "66": 65,
    "67": 66,
    "68": 67,
    "69": 68,
    "70": 69,
    "71": 70,
    "72": 71,
    "73": 72,
    "74": 73,
    "75": 74,
    "76": 75,
    "77": 76,
    "78": 77,
    "79": 78,
    "80": 79,
    "81": 80,
    "82": 81,
    "83": 82,
    "84": 83,
    "85": 84,
    "86": 85,
    "87": 86,
    "88": 87,
    "89": 88,
    "90": 89,
    "91": 90,
    "92": 91,
    "93": 92,
    "94": 93,
    "95": 94,
    "96": 95,
    "97": 96,
    "98": 97,
    "99": 98,
    "100": 99,
    "101": 100,
    "102": 101,
    "103": 102,
    "104": 103,
    "105": 104,
    "106": 105,
    "107": 106,
    "108": 107,
    "109": 108,
    "110": 109,
    "111": 110,
    "111A": 111,
    "111B": 112,
    "111C": 113,
    "111D": 114,
    "111E": 115,
    "111F": 116,
    "111G": 117,
    "111H": 118,
    "112I": 119,
    "112H": 120,
    "112G": 121,
    "112F": 122,
    "112E": 123,
    "112D": 124,
    "112C": 125,
    "112B": 126,
    "112A": 127,
    "112": 128,
    "113": 129,
    "114": 130,
    "115": 131,
    "116": 132,
    "117": 133,
    "118": 134,
    "119": 135,
    "120": 136,
    "121": 137,
    "122": 138,
    "123": 139,
    "124": 140,
    "125": 141,
    "126": 142,
    "127": 143,
    "128": 144,
}

L_dict = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "11": 10,
    "12": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "19": 18,
    "20": 19,
    "21": 20,
    "22": 21,
    "23": 22,
    "24": 23,
    "25": 24,
    "26": 25,
    "27": 26,
    "28": 27,
    "29": 28,
    "30": 29,
    "31": 30,
    "32": 31,
    "33": 32,
    "34": 33,
    "35": 34,
    "36": 35,
    "37": 36,
    "38": 37,
    "39": 38,
    "40": 39,
    "41": 40,
    "42": 41,
    "43": 42,
    "44": 43,
    "45": 44,
    "46": 45,
    "47": 46,
    "48": 47,
    "49": 48,
    "50": 49,
    "51": 50,
    "52": 51,
    "53": 52,
    "54": 53,
    "55": 54,
    "56": 55,
    "57": 56,
    "58": 57,
    "59": 58,
    "60": 59,
    "61": 60,
    "62": 61,
    "63": 62,
    "64": 63,
    "65": 64,
    "66": 65,
    "67": 66,
    "68": 67,
    "69": 68,
    "70": 69,
    "71": 70,
    "72": 71,
    "73": 72,
    "74": 73,
    "75": 74,
    "76": 75,
    "77": 76,
    "78": 77,
    "79": 78,
    "80": 79,
    "81": 80,
    "82": 81,
    "83": 82,
    "84": 83,
    "85": 84,
    "86": 85,
    "87": 86,
    "88": 87,
    "89": 88,
    "90": 89,
    "91": 90,
    "92": 91,
    "93": 92,
    "94": 93,
    "95": 94,
    "96": 95,
    "97": 96,
    "98": 97,
    "99": 98,
    "100": 99,
    "101": 100,
    "102": 101,
    "103": 102,
    "104": 103,
    "105": 104,
    "106": 105,
    "107": 106,
    "108": 107,
    "109": 108,
    "110": 109,
    "111": 110,
    "112": 111,
    "113": 112,
    "114": 113,
    "115": 114,
    "116": 115,
    "117": 116,
    "118": 117,
    "119": 118,
    "120": 119,
    "121": 120,
    "122": 121,
    "123": 122,
    "124": 123,
    "125": 124,
    "126": 125,
    "127": 126,
    "128": 127,
}

st.set_page_config(
    page_title="DeepSP App",
    layout="centered",
)

st.title("DeepSP")
st.header("Deep learning-based antibody structural properties")
st.subheader("The FASTA file format is H_seq/L_seq (variable regions)")

st.markdown("""
### EXAMPLE:
\>6p8n
QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMNWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGKNSDYNWDFQHWGQGTLVTVSS/DIVMSQSPSSLAVSVGEKVTMSCKSSQSLLYSSNQKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFTGSGSGTDFTLTISSVKAEDLAVYYCQQYEMFGGGTKLEIK 
""")

seq_file = st.file_uploader("#### Upload your FASTA file", type=["fasta"])
if seq_file is not None:
    stringio = StringIO(seq_file.getvalue().decode("utf-8"))
    sequences_H = []
    sequences_L = []
    name_list = []
    for record in SeqIO.parse(stringio, "fasta"):
        name = str(record.id)
        name_list.append(name)
        sequence = str(record.seq)
        sequence_H, sequence_L = sequence.split("/")
        sequences_H.append((name, sequence_H))
        sequences_L.append((name, sequence_L))

    results_H = anarci(sequences_H, scheme="imgt", output=False)
    results_L = anarci(sequences_L, scheme="imgt", output=False)
    numbering_H, alignment_details_H, hit_tables_H = results_H
    numbering_L, alignment_details_L, hit_tables_L = results_L

    # Iterate over the sequences
    seq_list = []
    for i in range(len(sequences_H)):
        if numbering_H[i] is None:
            print("ANARCI did not number", sequences_H[i][0])
        else:
            domain_numbering_H, start_index_H, end_index_H = numbering_H[i][0]
            domain_numbering_L, start_index_L, end_index_L = numbering_L[i][0]
            H_tmp = 145 * ["-"]
            L_tmp = 127 * ["-"]
            for j in range(len(domain_numbering_H)):
                col_H = str(domain_numbering_H[j][0][0]) + domain_numbering_H[j][0][1]
                col_H = col_H.replace(" ", "")
                H_tmp[H_dict[col_H]] = domain_numbering_H[j][1]
            for j in range(len(domain_numbering_L)):
                col_L = str(domain_numbering_L[j][0][0]) + domain_numbering_L[j][0][1]
                col_L = col_L.replace(" ", "")
                L_tmp[L_dict[col_L]] = domain_numbering_L[j][1]
            aa_string = ""
            for aa in H_tmp + L_tmp:
                aa_string += aa
            seq_list.append(aa_string)

    X = [one_hot_encoder(s=x) for x in seq_list]
    X = np.transpose(np.asarray(X), (0, 2, 1))
    X = np.asarray(X)

    # load DeepSAP_pos model
    json_file = open("../data/Conv1D_regressionSAPpos.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into the model
    loaded_model.load_weights("../data/Conv1D_regression_SAPpos.h5")
    loaded_model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    # predict SAPpos
    y_pred = loaded_model.predict(X)
    df_SAPpos = pd.DataFrame(
        y_pred,
        columns=[
            "SAP_pos_CDRH1",
            "SAP_pos_CDRH2",
            "SAP_pos_CDRH3",
            "SAP_pos_CDRL1",
            "SAP_pos_CDRL2",
            "SAP_pos_CDRL3",
            "SAP_pos_CDR",
            "SAP_pos_Hv",
            "SAP_pos_Lv",
            "SAP_pos_Fv",
        ],
    )

    # load DeepSCM_neg model
    json_file = open("../data/Conv1D_regressionSCMneg.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    # load weights into the model
    loaded_model.load_weights("../data/Conv1D_regression_SCMneg.h5")
    loaded_model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    # predict SCMneg
    y_pred = loaded_model.predict(X)
    df_SCMneg = pd.DataFrame(
        y_pred,
        columns=[
            "SCM_neg_CDRH1",
            "SCM_neg_CDRH2",
            "SCM_neg_CDRH3",
            "SCM_neg_CDRL1",
            "SCM_neg_CDRL2",
            "SCM_neg_CDRL3",
            "SCM_neg_CDR",
            "SCM_neg_Hv",
            "SCM_neg_Lv",
            "SCM_neg_Fv",
        ],
    )

    # load DeepSCM_pos model
    json_file = open("../data/Conv1D_regressionSCMpos.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into the model
    loaded_model.load_weights("../data/Conv1D_regression_SCMpos.h5")
    loaded_model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    # predict SCMpos
    y_pred = loaded_model.predict(X)
    df_SCMpos = pd.DataFrame(
        y_pred,
        columns=[
            "SCM_pos_CDRH1",
            "SCM_pos_CDRH2",
            "SCM_pos_CDRH3",
            "SCM_pos_CDRL1",
            "SCM_pos_CDRL2",
            "SCM_pos_CDRL3",
            "SCM_pos_CDR",
            "SCM_pos_Hv",
            "SCM_pos_Lv",
            "SCM_pos_Fv",
        ],
    )

    df_name = pd.DataFrame(name_list, columns=["ID"])

    df_DeepSP = pd.concat([df_name, df_SAPpos, df_SCMneg, df_SCMpos], axis=1)
    st.dataframe(data=df_DeepSP, use_container_width=True, hide_index=True)
