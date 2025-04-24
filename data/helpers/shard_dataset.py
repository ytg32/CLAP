import os
import tarfile
import json
import math
import argparse
from pathlib import Path

def create_shards(input_dirs, output_dir, samples_per_shard):
    os.makedirs(output_dir, exist_ok=True)
    sizes_file = os.path.join(output_dir, "sizes.json")

    samples = []

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        wav_files = {
            f.relative_to(input_path).with_suffix(""): f
            for f in input_path.glob("**/*.wav")
        }

        json_files = {
            f.relative_to(input_path).with_suffix(""): f
            for f in input_path.glob("**/*.json")
        }

        common_keys = wav_files.keys() & json_files.keys()

        for key in common_keys:
            samples.append((wav_files[key], json_files[key]))

    total_samples = len(samples)
    num_shards = math.ceil(total_samples / samples_per_shard)

    print(f"Total samples: {total_samples}")
    print(f"Total shards to be created: {num_shards}")

    shard_sizes = {}
    shard_count = 0

    for i in range(0, total_samples, samples_per_shard):
        shard_samples = samples[i:i + samples_per_shard]
        shard_name = f"{shard_count:05d}.tar"
        shard_path = os.path.join(output_dir, shard_name)

        with tarfile.open(shard_path, "w") as tar:
            for wav_path, json_path in shard_samples:
                tar.add(wav_path, arcname=wav_path.name)
                tar.add(json_path, arcname=json_path.name)

        shard_sizes[shard_name] = len(shard_samples)
        print(f"Created {shard_path} with {len(shard_samples)} samples")
        shard_count += 1

    sizes_data = {
        "num_shards": num_shards,
        "shards": shard_sizes
    }

    with open(sizes_file, "w") as f:
        json.dump(sizes_data, f, indent=4)

    print(f"Completed! {num_shards} shards created in {output_dir}. Sizes saved to {sizes_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a dataset into tar files for WebDataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True, nargs='+',
        help="Path(s) to the input dataset folder(s)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the output folder for tar shards"
    )
    parser.add_argument(
        "--samples_per_shard", type=int, default=1024,
        help="Number of samples (json+wav pairs) per shard"
    )

    args = parser.parse_args()
    create_shards(args.input_dir, args.output_dir, args.samples_per_shard)
