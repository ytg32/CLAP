import os
import argparse

def check_pairs(directory):
    json_files = set()
    wav_files = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.add(os.path.splitext(file)[0])
            elif file.endswith('.wav'):
                wav_files.add(os.path.splitext(file)[0])

    missing_wav = json_files - wav_files
    missing_json = wav_files - json_files

    if missing_wav:
        print(f"❌ Missing .wav files for these .jsons:\n  {', '.join(sorted(missing_wav))}")
    else:
        print("✅ All .json files have corresponding .wav files.")

    if missing_json:
        print(f"❌ Missing .json files for these .wavs:\n  {', '.join(sorted(missing_json))}")
    else:
        print("✅ All .wav files have corresponding .json files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check that all .json and .wav files in a folder are paired.")
    parser.add_argument("directory", type=str, help="Path to the directory to check")
    args = parser.parse_args()

    check_pairs(args.directory)
