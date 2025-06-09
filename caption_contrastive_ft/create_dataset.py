import json
import csv
from pathlib import Path

def collect_species_names(data_dir, output_csv_path, languages):
    """
    Traverses all JSON files under `data_dir` and writes a combined CSV with
    scientific names and common names in specified languages.
    Skips species that don't have all the requested languages.
    Synonyms are ignored.

    :param data_dir: Root directory containing JSON species files
    :param output_csv_path: Output CSV path
    :param languages: List of language codes to extract (e.g., ['en', 'fr', 'de'])
    """
    headers = ['scientific_name'] + languages
    rows = []

    for json_file in Path(data_dir).rglob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                species_list = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping {json_file}: invalid JSON ({e})")
                continue

        for species in species_list:
            row = {'scientific_name': species.get('scientific_name', '')}
            all_present = True

            for lang in languages:
                name_entry = ''
                for entry in species.get('common_names', []):
                    if entry.get('lang') == lang:
                        name_entry = entry['name']
                        break
                if not name_entry:
                    all_present = False
                    break
                row[lang] = name_entry

            if all_present:
                rows.append([row[h] for h in headers])

    # Write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# Example usage:
collect_species_names(
    data_dir='dataset/data/Vertebrata',
    output_csv_path='species_names_filtered.csv',
    languages=['en']
)
