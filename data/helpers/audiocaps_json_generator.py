import pandas as pd
import json
import os
import argparse

def main(tsv_path, output_dir):
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by 'uniq_id' to combine captions for each audio file
    grouped = df.groupby('uniq_id')['text'].apply(list).reset_index()

    # Iterate through the grouped data and create JSON files
    for idx, row in grouped.iterrows():
        audio_id = str(row['uniq_id']).split('/')[-1].replace('.flac', '')  # Get the file name without extension
        captions = row['text']

        # Prepare JSON data with all captions as a list
        json_data = {"text": captions}

        # Define JSON file path
        json_file_path = os.path.join(output_dir, f"{audio_id}.json")

        # Write the JSON data to the file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    print(f"✅ JSON files created successfully at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an AudioCaps-style TSV file to per-audio JSON caption files."
    )
    parser.add_argument(
        "--tsv_path", "-t", 
        type=str, 
        required=True, 
        help="Path to the input TSV file (e.g., audiocaps_train.tsv)"
    )
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        required=True, 
        help="Directory to save the generated JSON files"
    )

    args = parser.parse_args()
    main(args.tsv_path, args.output_dir)
