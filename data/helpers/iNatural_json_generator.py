import os
import json
import requests
import random
import argparse

def get_common_name(scientific_name):
    """Fetch the common name from GBIF API."""
    url = f"https://api.gbif.org/v1/species/match?name={scientific_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        common_name = data.get("vernacularName")
        return common_name if common_name else ""
    return ""

def process_directory(base_path):
    """Iterate through directories, process WAV files, and create JSON metadata."""
    for root, _, files in os.walk(base_path):
        folder_name = os.path.basename(root)
        parts = folder_name.split("_")
        if len(parts) < 2:
            continue  # Skip folders without expected naming

        scientific_name = f"{parts[-2]} {parts[-1]}"
        full_taxonomy = " ".join(parts[1:])

        if random.choice([True, False]):
            full_taxonomy = f"sound of {full_taxonomy}"
        if random.choice([True, False]):
            scientific_name = f"sound of {scientific_name}"

        common_name = get_common_name(scientific_name)

        for file in files:
            if file.endswith(".wav"):
                json_filename = os.path.splitext(file)[0] + ".json"
                json_path = os.path.join(root, json_filename)

                data = {"text": [full_taxonomy, scientific_name]}
                if common_name:
                    data["text"].append(common_name)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                print(f"✅ Created: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON metadata for WAV files with taxonomy labels.")
    parser.add_argument(
        "--data_path", "-d", 
        type=str, 
        required=True, 
        help="Path to the dataset directory containing folders named by taxonomy (e.g., 'iNatural/train')"
    )
    args = parser.parse_args()
    process_directory(args.data_path)
