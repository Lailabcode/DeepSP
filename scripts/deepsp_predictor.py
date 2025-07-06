# Import libraries
import os
import subprocess # For running ANARCI
from pathlib import Path # For path management
import shutil # For removing temporary directories

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Should be set before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
import pandas as pd

# Import machine learning libraries
# from tensorflow.keras.models import model_from_json # No longer directly used
# from keras.models import model_from_json # No longer directly used

from Bio import SeqIO
# from Bio.Seq import Seq # Not directly used
# from Bio.SeqRecord import SeqRecord # Not directly used

# Import from our utility script
from utils import (
    one_hot_encoder, # Will be used after ANARCI processing
    parse_anarci_results_to_aligned_sequences, # Replaces old seq_preprocessing
    predict_properties,
    get_project_root,
    dataframe_to_fasta_string, # To create FASTA string for ANARCI input
)

def main():
    project_root = get_project_root()
    data_dir = project_root / "data"
    input_csv_path = data_dir / "input" / "DeepSP_input.csv" # Corrected path
    output_csv_path = data_dir / "DeepSP_descriptors.csv"

    # Temporary directory for ANARCI files
    # Using a subdirectory within scripts for temporary files, or use tempfile module
    temp_anarci_dir = project_root / "scripts" / "temp_anarci_processing"
    temp_anarci_dir.mkdir(exist_ok=True)

    heavy_fasta_path = temp_anarci_dir / "seq_H.fasta"
    light_fasta_path = temp_anarci_dir / "seq_L.fasta"
    anarci_H_csv_path = temp_anarci_dir / "seq_aligned_H.csv" # ANARCI default output name based on prefix
    anarci_L_csv_path = temp_anarci_dir / "seq_aligned_KL.csv" # ANARCI default output name based on prefix

    try:
        # Import dataset
        try:
            dataset = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_csv_path}")
            return

        # Create FASTA strings from DataFrame columns
        fasta_H_str = dataframe_to_fasta_string(dataset, "Name", "Heavy_Chain", "Light_Chain").split('\n')
        # The dataframe_to_fasta_string creates combined H/L, need to adapt or split
        # For simplicity, let's recreate separate FASTA strings for H and L

        with open(heavy_fasta_path, "w") as h_out, open(light_fasta_path, "w") as l_out:
            for _, row in dataset.iterrows():
                h_out.write(f">{row['Name']}\n{row['Heavy_Chain']}\n")
                l_out.write(f">{row['Name']}\n{row['Light_Chain']}\n")

        # Sequence alignment with ANARCI using subprocess
        anarci_cmd_H = [
            "ANARCI", "-i", str(heavy_fasta_path), "-o", str(temp_anarci_dir / "seq_aligned"),
            "-s", "imgt", "-r", "heavy", "--csv"
        ]
        anarci_cmd_L = [
            "ANARCI", "-i", str(light_fasta_path), "-o", str(temp_anarci_dir / "seq_aligned"),
            "-s", "imgt", "-r", "light", "--csv"
        ]

        print(f"Running ANARCI for Heavy Chains: {' '.join(anarci_cmd_H)}")
        process_H = subprocess.run(anarci_cmd_H, capture_output=True, text=True)
        if process_H.returncode != 0:
            print(f"ANARCI Error (Heavy Chains):\n{process_H.stderr}")
            # Optionally, raise an error or handle it
        else:
            print("ANARCI Heavy Chain processing successful.")
            # print(process_H.stdout) # For debugging ANARCI output

        print(f"Running ANARCI for Light Chains: {' '.join(anarci_cmd_L)}")
        process_L = subprocess.run(anarci_cmd_L, capture_output=True, text=True)
        if process_L.returncode != 0:
            print(f"ANARCI Error (Light Chains):\n{process_L.stderr}")
            # Optionally, raise an error or handle it
        else:
            print("ANARCI Light Chain processing successful.")
            # print(process_L.stdout) # For debugging ANARCI output

        # Check if ANARCI output files were created
        if not anarci_H_csv_path.exists() or not anarci_L_csv_path.exists():
            print("Error: ANARCI did not produce the expected output CSV files.")
            print(f"Checked for: {anarci_H_csv_path} and {anarci_L_csv_path}")
            # Fallback: try reading the _H.csv and _KL.csv directly if ANARCI output them without prefix path
            # This part is a bit of a guess if ANARCI doesn't write to the full path specified by -o
            if Path("seq_aligned_H.csv").exists() and Path("seq_aligned_KL.csv").exists():
                 print("Found ANARCI output in current directory. Moving them.")
                 shutil.move("seq_aligned_H.csv", anarci_H_csv_path)
                 shutil.move("seq_aligned_KL.csv", anarci_L_csv_path)
            else:
                 return


        # Sequence preprocessing using the utility function
        # This function replaces the old seq_preprocessing() and load_input_data("seq_aligned_HL.txt")
        valid_name_list, aligned_seq_list = parse_anarci_results_to_aligned_sequences(
            str(anarci_H_csv_path), str(anarci_L_csv_path)
        )

        if not aligned_seq_list:
            print("No sequences were successfully aligned by ANARCI or parsed.")
            return

        # One Hot Encoding of Aligned Sequence
        X_one_hot = [one_hot_encoder(s=x) for x in aligned_seq_list]
        X_processed = np.transpose(np.asarray(X_one_hot), (0, 2, 1))
        X_processed = np.asarray(X_processed)

        if X_processed.size == 0:
            print("Data processing resulted in an empty array. Cannot predict.")
            return

        # Predict DeepSP Descriptors using the utility function
        # The model_base_path will default to project_root / "data" in predict_properties
        df_predictions = predict_properties(X_processed, model_base_path=str(data_dir))

        if df_predictions.empty:
            print("Predictions resulted in an empty DataFrame.")
            return

        df_name = pd.DataFrame(valid_name_list, columns=["Name"])

        # Concatenate names with predictions
        df_final = pd.concat([df_name.reset_index(drop=True), df_predictions.reset_index(drop=True)], axis=1)

        # Save the final DataFrame
        df_final.to_csv(output_csv_path, index=False)
        print(f"Successfully saved DeepSP descriptors to {output_csv_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
    finally:
        # Clean up temporary files and directory
        if temp_anarci_dir.exists():
            try:
                shutil.rmtree(temp_anarci_dir)
                print(f"Successfully removed temporary directory: {temp_anarci_dir}")
            except Exception as e:
                print(f"Error removing temporary directory {temp_anarci_dir}: {e}")

if __name__ == "__main__":
    main()
