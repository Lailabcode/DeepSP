# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from anarci import anarci # anarci 임포트는 DeepSP-app.py 에서 확인되어 추가합니다.
from Bio import SeqIO # Bio 임포트는 DeepSP-app.py 에서 확인되어 추가합니다.
from io import StringIO # StringIO 임포트는 DeepSP-app.py 에서 확인되어 추가합니다.
from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

# Constants for ANARCI processing
H_inclusion_list = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
    "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100",
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "111A", "111B", "111C", "111D", "111E",
    "111F", "111G", "111H", "112I", "112H", "112G", "112F", "112E", "112D", "112C", "112B", "112A", "112", "113", "114",
    "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128"
]

L_inclusion_list = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
    "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100",
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117",
    "118", "119", "120", "121", "122", "123", "124", "125", "126", "127"
]

H_dict = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11, "13": 12, "14": 13,
    "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "26": 25,
    "27": 26, "28": 27, "29": 28, "30": 29, "31": 30, "32": 31, "33": 32, "34": 33, "35": 34, "36": 35, "37": 36, "38": 37,
    "39": 38, "40": 39, "41": 40, "42": 41, "43": 42, "44": 43, "45": 44, "46": 45, "47": 46, "48": 47, "49": 48, "50": 49,
    "51": 50, "52": 51, "53": 52, "54": 53, "55": 54, "56": 55, "57": 56, "58": 57, "59": 58, "60": 59, "61": 60, "62": 61,
    "63": 62, "64": 63, "65": 64, "66": 65, "67": 66, "68": 67, "69": 68, "70": 69, "71": 70, "72": 71, "73": 72, "74": 73,
    "75": 74, "76": 75, "77": 76, "78": 77, "79": 78, "80": 79, "81": 80, "82": 81, "83": 82, "84": 83, "85": 84, "86": 85,
    "87": 86, "88": 87, "89": 88, "90": 89, "91": 90, "92": 91, "93": 92, "94": 93, "95": 94, "96": 95, "97": 96, "98": 97,
    "99": 98, "100": 99, "101": 100, "102": 101, "103": 102, "104": 103, "105": 104, "106": 105, "107": 106, "108": 107,
    "109": 108, "110": 109, "111": 110, "111A": 111, "111B": 112, "111C": 113, "111D": 114, "111E": 115, "111F": 116,
    "111G": 117, "111H": 118, "112I": 119, "112H": 120, "112G": 121, "112F": 122, "112E": 123, "112D": 124, "112C": 125,
    "112B": 126, "112A": 127, "112": 128, "113": 129, "114": 130, "115": 131, "116": 132, "117": 133, "118": 134,
    "119": 135, "120": 136, "121": 137, "122": 138, "123": 139, "124": 140, "125": 141, "126": 142, "127": 143, "128": 144
}

L_dict = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11, "13": 12, "14": 13,
    "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "26": 25,
    "27": 26, "28": 27, "29": 28, "30": 29, "31": 30, "32": 31, "33": 32, "34": 33, "35": 34, "36": 35, "37": 36, "38": 37,
    "39": 38, "40": 39, "41": 40, "42": 41, "43": 42, "44": 43, "45": 44, "46": 45, "47": 46, "48": 47, "49": 48, "50": 49,
    "51": 50, "52": 51, "53": 52, "54": 53, "55": 54, "56": 55, "57": 56, "58": 57, "59": 58, "60": 59, "61": 60, "62": 61,
    "63": 62, "64": 63, "65": 64, "66": 65, "67": 66, "68": 67, "69": 68, "70": 69, "71": 70, "72": 71, "73": 72, "74": 73,
    "75": 74, "76": 75, "77": 76, "78": 77, "79": 78, "80": 79, "81": 80, "82": 81, "83": 82, "84": 83, "85": 84, "86": 85,
    "87": 86, "88": 87, "89": 88, "90": 89, "91": 90, "92": 91, "93": 92, "94": 93, "95": 94, "96": 95, "97": 96, "98": 97,
    "99": 98, "100": 99, "101": 100, "102": 101, "103": 102, "104": 103, "105": 104, "106": 105, "107": 106, "108": 107,
    "109": 108, "110": 109, "111": 110, "112": 111, "113": 112, "114": 113, "115": 114, "116": 115, "117": 116, "118": 117,
    "119": 118, "120": 119, "121": 120, "122": 121, "123": 122, "124": 123, "125": 124, "126": 125, "127": 126, "128": 127
}


def one_hot_encoder(s: str) -> np.ndarray:
    """
    Encodes a sequence string into a one-hot numpy array.

    Args:
        s: The input sequence string.

    Returns:
        A numpy array representing the one-hot encoded sequence.
    """
    d = {
        "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
        "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18,
        "Y": 19, "-": 20,
    }
    x = np.zeros((len(d), len(s)))
    try:
        x[[d[c] for c in s], range(len(s))] = 1
    except KeyError as e:
        raise ValueError(f"Invalid character in sequence: {e}. Only ACDEFGHIKLMNPQRSTVWY- are allowed.")
    return x


def load_and_preprocess_data_from_fasta_input(
    sequences_H: list[tuple[str, str]],
    sequences_L: list[tuple[str, str]]
) -> tuple[list[str], np.ndarray]:
    """
    Processes FASTA input using ANARCI, performs one-hot encoding.
    This function is adapted from DeepSP-app.py and deepsp_predictor.py.

    Args:
        sequences_H: A list of tuples, where each tuple contains the name and heavy chain sequence.
        sequences_L: A list of tuples, where each tuple contains the name and light chain sequence.

    Returns:
        A tuple containing:
            - name_list: List of sequence names.
            - X_processed: Processed and one-hot encoded sequences as a numpy array.
    """
    name_list = [seq_info[0] for seq_info in sequences_H] # Assuming names are consistent

    results_H = anarci(sequences_H, scheme="imgt", output=False)
    results_L = anarci(sequences_L, scheme="imgt", output=False)
    numbering_H, _, _ = results_H
    numbering_L, _, _ = results_L

    seq_list_aligned = []
    valid_name_list = []

    for i in range(len(sequences_H)):
        if numbering_H[i] is None or numbering_L[i] is None:
            print(f"ANARCI did not number {sequences_H[i][0]} for H chain or {sequences_L[i][0]} for L chain. Skipping.")
            continue

        valid_name_list.append(sequences_H[i][0])
        domain_numbering_H, _, _ = numbering_H[i][0]
        domain_numbering_L, _, _ = numbering_L[i][0]

        H_tmp = 145 * ["-"]
        L_tmp = 127 * ["-"]

        for j in range(len(domain_numbering_H)):
            col_H = str(domain_numbering_H[j][0][0]) + domain_numbering_H[j][0][1]
            col_H = col_H.replace(" ", "")
            if col_H in H_dict:
                 H_tmp[H_dict[col_H]] = domain_numbering_H[j][1]
            # else:
                # print(f"Warning: Position {col_H} not in H_dict for sequence {sequences_H[i][0]}")


        for j in range(len(domain_numbering_L)):
            col_L = str(domain_numbering_L[j][0][0]) + domain_numbering_L[j][0][1]
            col_L = col_L.replace(" ", "")
            if col_L in L_dict:
                L_tmp[L_dict[col_L]] = domain_numbering_L[j][1]
            # else:
                # print(f"Warning: Position {col_L} not in L_dict for sequence {sequences_L[i][0]}")

        aa_string = "".join(H_tmp + L_tmp)
        seq_list_aligned.append(aa_string)

    if not seq_list_aligned:
        return [], np.array([])

    X_one_hot = [one_hot_encoder(s=x) for x in seq_list_aligned]
    X_processed = np.transpose(np.asarray(X_one_hot), (0, 2, 1))
    X_processed = np.asarray(X_processed)

    return valid_name_list, X_processed


def load_model_and_weights(model_json_path: str, model_weights_path: str):
    """
    Loads a Keras model from a JSON file and its weights.

    Args:
        model_json_path: Path to the model JSON file.
        model_weights_path: Path to the model weights H5 file.

    Returns:
        A compiled Keras model.
    """
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    loaded_model.compile(optimizer="adam", loss="mae", metrics=["mae"]) # Default compile, can be overridden
    return loaded_model

def predict_properties(X_processed: np.ndarray, model_base_path: str) -> pd.DataFrame:
    """
    Predicts SAPpos, SCMneg, and SCMpos properties using pre-trained models.

    Args:
        X_processed: Processed and one-hot encoded input data.
        model_base_path: Base path to the directory containing model files.
                         If None, defaults to `get_project_root() / "data"`.

    Returns:
        A pandas DataFrame containing the predicted properties.
                         The DataFrame will have columns for each predicted property.
    """
    if X_processed.size == 0:
        return pd.DataFrame()

    if model_base_path is None:
        model_base_path = get_project_root() / "data"
    else:
        model_base_path = Path(model_base_path)

    # SAPpos
    sap_pos_model = load_model_and_weights(
        model_base_path / "Conv1D_regressionSAPpos.json",
        model_base_path / "Conv1D_regression_SAPpos.h5"
    )
    sap_pos_pred = sap_pos_model.predict(X_processed)
    df_sap_pos = pd.DataFrame(sap_pos_pred, columns=[
        "SAP_pos_CDRH1", "SAP_pos_CDRH2", "SAP_pos_CDRH3", "SAP_pos_CDRL1", "SAP_pos_CDRL2",
        "SAP_pos_CDRL3", "SAP_pos_CDR", "SAP_pos_Hv", "SAP_pos_Lv", "SAP_pos_Fv"
    ])

    # SCMneg
    scm_neg_model = load_model_and_weights(
        model_base_path / "Conv1D_regressionSCMneg.json",
        model_base_path / "Conv1D_regression_SCMneg.h5"
    )
    scm_neg_pred = scm_neg_model.predict(X_processed)
    df_scm_neg = pd.DataFrame(scm_neg_pred, columns=[
        "SCM_neg_CDRH1", "SCM_neg_CDRH2", "SCM_neg_CDRH3", "SCM_neg_CDRL1", "SCM_neg_CDRL2",
        "SCM_neg_CDRL3", "SCM_neg_CDR", "SCM_neg_Hv", "SCM_neg_Lv", "SCM_neg_Fv"
    ])

    # SCMpos
    scm_pos_model = load_model_and_weights(
        model_base_path / "Conv1D_regressionSCMpos.json",
        model_base_path / "Conv1D_regression_SCMpos.h5"
    )
    scm_pos_pred = scm_pos_model.predict(X_processed)
    df_scm_pos = pd.DataFrame(scm_pos_pred, columns=[
        "SCM_pos_CDRH1", "SCM_pos_CDRH2", "SCM_pos_CDRH3", "SCM_pos_CDRL1", "SCM_pos_CDRL2",
        "SCM_pos_CDRL3", "SCM_pos_CDR", "SCM_pos_Hv", "SCM_pos_Lv", "SCM_pos_Fv"
    ])

    return pd.concat([df_sap_pos, df_scm_neg, df_scm_pos], axis=1)


def load_input_data_for_training(filename: Path | str) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Loads input data from a file for training purposes.
    Adapted from DeepSP_model_train.py

    Args:
        filename: Path to the data file.

    Returns:
        A tuple containing:
            - name_list: List of sequence names.
            - seq_list: List of sequences.
            - score_list: List of scores.
    """
    name_list = []
    seq_list = []
    score_list = []

    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
            score_temp = []
            for i in range(len(line[2:])):
                data = float(line[i + 2])
                score_temp.append(data)
            score_list.append(score_temp)
    return name_list, seq_list, score_list

# The seq_preprocessing function from deepsp_predictor.py and notebooks/DeepSP_predictor.ipynb
# is more complex as it involves reading and writing intermediate files (seq_aligned_H.csv, etc.).
# It's better to refactor the calling scripts to use the ANARCI directly via the `anarci` library
# and then use the `load_and_preprocess_data_from_fasta_input` which takes sequence data as input.
# However, if the exact logic of `seq_preprocessing` (including file I/O) is needed, it can be added here.
# For now, I will assume that the calling scripts will be refactored to avoid this file-based intermediate step.

# Example of how ANARCI can be called from FASTA file (similar to deepsp_predictor.py)
# This is more for illustration as the `load_and_preprocess_data_from_fasta_input` handles direct sequence input.
def run_anarci_on_fasta_files(heavy_fasta_path: str, light_fasta_path: str, output_prefix:str = "seq_aligned"):
    """
    Runs ANARCI on heavy and light chain FASTA files.
    This is more of a conceptual function as direct sequence processing is preferred.
    """
    import os
    # This would typically use subprocess for better control
    os.system(f"ANARCI -i {heavy_fasta_path} -o {output_prefix} -s imgt -r heavy --csv")
    os.system(f"ANARCI -i {light_fasta_path} -o {output_prefix} -s imgt -r light --csv")

    # Results would be in {output_prefix}_H.csv and {output_prefix}_KL.csv

def parse_anarci_results_to_aligned_sequences(
    anarci_H_csv_path: str,
    anarci_L_csv_path: str
) -> tuple[list[str], list[str]]:
    """
    Parses ANARCI output CSVs and generates aligned sequences.
    This function replicates the core logic of 'seq_preprocessing' from the predictor scripts.

    Args:
        anarci_H_csv_path: Path to the ANARCI output CSV for heavy chains.
        anarci_L_csv_path: Path to the ANARCI output CSV for light chains.

    Returns:
        A tuple containing:
            - name_list: List of sequence IDs.
            - aligned_seq_list: List of combined and aligned HL sequences.
    """
    infile_H = pd.read_csv(anarci_H_csv_path)
    infile_L = pd.read_csv(anarci_L_csv_path)

    name_list = []
    aligned_seq_list = []

    # Assuming infile_H and infile_L have the same number of rows and corresponding IDs
    # This might need error handling if IDs don't match or rows are missing
    num_mAbs = len(infile_H["Id"])
    if len(infile_L["Id"]) != num_mAbs:
        raise ValueError("Heavy and Light chain ANARCI output files have different number of entries.")

    for i in range(num_mAbs):
        current_id_H = infile_H.iloc[i]["Id"]
        # Find corresponding Light chain entry (this assumes order or requires matching)
        # For simplicity, assuming they are in the same order. A robust solution would match by Id.
        current_id_L = infile_L.iloc[i]["Id"]
        if current_id_H != current_id_L:
             # Attempt to find by ID if order is not guaranteed
            l_row_series = infile_L[infile_L["Id"] == current_id_H]
            if l_row_series.empty:
                print(f"Warning: Could not find matching Light chain for Heavy chain ID {current_id_H}. Skipping.")
                continue
            l_row = l_row_series.iloc[0]
        else:
            l_row = infile_L.iloc[i]


        name_list.append(current_id_H)

        H_tmp = 145 * ["-"]
        L_tmp = 127 * ["-"]

        for col_name, aa_val in infile_H.iloc[i].items():
            if col_name in H_inclusion_list and pd.notna(aa_val):
                H_tmp[H_dict[col_name]] = aa_val

        for col_name, aa_val in l_row.items():
            if col_name in L_inclusion_list and pd.notna(aa_val):
                L_tmp[L_dict[col_name]] = aa_val

        aligned_seq_list.append("".join(H_tmp + L_tmp))

    return name_list, aligned_seq_list

def convert_sequences_to_fasta_string(sequences: list[tuple[str,str]]) -> str:
    """
    Converts a list of (name, sequence) tuples to a FASTA formatted string.
    """
    fasta_string = ""
    for name, seq in sequences:
        fasta_string += f">{name}\n{seq}\n"
    return fasta_string

def dataframe_to_fasta_string(df: pd.DataFrame, id_col: str, heavy_col: str, light_col: str) -> str:
    """
    Converts a DataFrame with ID, Heavy Chain, Light Chain columns to a FASTA formatted string.
    The sequence for each entry will be "heavy_sequence/light_sequence".
    """
    fasta_str = ""
    for _, row in df.iterrows():
        fasta_str += f">{row[id_col]}\n{row[heavy_col]}/{row[light_col]}\n"
    return fasta_str
