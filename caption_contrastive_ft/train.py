import os
import json
import random
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer, losses
from datasets import Dataset
import pickle  # Use pickle to save/load data
from tqdm import tqdm

class CaptionDataProcessor:
    def __init__(self, dataset_paths, audiocaps_path, negative_sampling_ratio=0.2, save_dir="processed_data"):
        self.dataset_paths = dataset_paths
        self.audiocaps_path = audiocaps_path
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir

        # Initialize storage
        self.class_to_captions = defaultdict(set)  # for train
        self.audiocaps_captions = set()  # for negative sampling in train
        self.class_to_captions_val = defaultdict(set)  # for validation
        self.audiocaps_captions_val = set()  # for negative sampling in validation

        # Make sure save_dir exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load_data(self, force_reload=False):
        """Load data from the provided paths or from saved files."""
        if not force_reload:
            # Try to load from saved data
            if self._load_saved_data("train") and self._load_saved_data("val"):
                return  # Data already loaded from files

        # Otherwise, process and save the data
        self._process_data("train")
        self._process_data("val")
        self._save_data("train")
        self._save_data("val")

    def _process_data(self, mode):
        """Process data for both animal_speak, iNatural, and audiocaps datasets for the given mode."""
        # Process animal_speak and iNatural datasets
        if mode == "train":
            dataset_paths = self.dataset_paths["train"]
        else:
            dataset_paths = self.dataset_paths["val"]

        # Process animal_speak and iNatural data
        for dataset_root in dataset_paths:
            file_count = sum(len(files) for _, _, files in os.walk(dataset_root))
            with tqdm(total=file_count, desc=f"Processing {dataset_root} ({mode})") as pbar: 
                for folder, _, files in os.walk(dataset_root):
                    class_name = os.path.basename(folder)
                    for file in files: 
                        pbar.update(1)
                        if file.endswith(".json"):
                            self._load_json_file(os.path.join(folder, file), class_name, is_val=(mode == "val"))

        # Process audiocaps data
        audiocaps_path = self.audiocaps_path["train"] if mode == "train" else self.audiocaps_path["val"]
        for _, _, files in os.walk(audiocaps_path):
            for file in tqdm(files, desc=f"Processing Audiocaps {mode}"):  # Adding tqdm for progress
                if file.endswith(".json"):
                    file_path = os.path.join(audiocaps_path, file)
                    self._load_audiocaps_json(file_path, is_val=(mode == "val"))

    def _load_json_file(self, file_path, class_name, is_val):
        """Helper function to load and process individual JSON files for animal_speak and iNAtural."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if "text" in data:
                    # Even if there's just one caption, we allow it for animal_speak and iNAtural
                    target_dict = self.class_to_captions_val if is_val else self.class_to_captions
                    target_dict[class_name].update(data["text"])
        except Exception as e:
            print(f"⚠️ Error in {file_path}: {e}")

    def _load_audiocaps_json(self, file_path, is_val):
        """Helper function to load and process audiocaps JSON files."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if "text" in data:
                    # Audiocaps should have multiple captions (more than one)
                    target_set = self.audiocaps_captions_val if is_val else self.audiocaps_captions
                    target_set.update(data["text"])
                else:
                    print(f"⚠️ Skipping {file_path}: Audiocaps should contain more than one caption.")
        except Exception as e:
            print(f"⚠️ Error in {file_path}: {e}")

    def _save_data(self, mode):
        """Save the processed data to files (pickle format)."""
        file_path = os.path.join(self.save_dir, f"{mode}_data.pkl")
        data = {
            "class_to_captions": self.class_to_captions,
            "audiocaps_captions": self.audiocaps_captions,
            "class_to_captions_val": self.class_to_captions_val,
            "audiocaps_captions_val": self.audiocaps_captions_val
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_saved_data(self, mode):
        """Load preprocessed data from saved files (pickle format)."""
        file_path = os.path.join(self.save_dir, f"{mode}_data.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.class_to_captions = data["class_to_captions"]
                self.audiocaps_captions = data["audiocaps_captions"]
                self.class_to_captions_val = data["class_to_captions_val"]
                self.audiocaps_captions_val = data["audiocaps_captions_val"]
            return True
        return False

    def build_pairs(self, is_val=False):
        """Generate pairs (positive and negative) for both training and validation."""
        class_to_captions = self.class_to_captions_val if is_val else self.class_to_captions
        audiocaps_captions = self.audiocaps_captions_val if is_val else self.audiocaps_captions

        sentence1_list, sentence2_list, label_list = [], [], []

        # Positive pairs: all combinations within a class
        for class_name, captions in class_to_captions.items():
            if len(captions) < 2:
                continue
            for s1, s2 in combinations(captions, 2):
                sentence1_list.append(s1)
                sentence2_list.append(s2)
                label_list.append(1)

        # Negative pairs: sample from different classes
        classes = list(class_to_captions.keys())
        num_negatives = len(label_list)

        for _ in range(num_negatives):
            if random.random() < self.negative_sampling_ratio:
                s1 = random.choice(list(class_to_captions[random.choice(classes)]))
                s2 = random.choice(list(audiocaps_captions))
            else:
                c1, c2 = random.sample(classes, 2)
                s1 = random.choice(list(class_to_captions[c1]))
                s2 = random.choice(list(class_to_captions[c2]))

            sentence1_list.append(s1)
            sentence2_list.append(s2)
            label_list.append(0)

        return sentence1_list, sentence2_list, label_list

    def get_batches(self, batch_size=32, is_val=False):
        """Yields batches of (sentence1, sentence2, label)."""
        s1_list, s2_list, labels = self.build_pairs(is_val=is_val)
        total = len(labels)
    
        for i in range(0, total, batch_size):
            yield (
                s1_list[i:i+batch_size],
                s2_list[i:i+batch_size],
                labels[i:i+batch_size]
            )
 
    def create_dataset(self, sentence1_list, sentence2_list, label_list):
        """Create a dataset from the sentence pairs and labels."""
        return Dataset.from_dict({
            "sentence1": sentence1_list,
            "sentence2": sentence2_list,
            "label": label_list,
        })


class ContrastiveModelTrainer:
    def __init__(self, model_name, train_dataset, val_dataset):
        # Change the model name here to use ModernBERT
        self.model = SentenceTransformer(model_name)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss = losses.ContrastiveLoss(self.model)

    def train(self):
        """Train the model."""
        trainer = SentenceTransformerTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            evaluator=self.val_dataset,
            loss=self.loss
        )
        trainer.train()


# Usage Example:
dataset_paths = {
    "train": [
        "/cluster/work/boraa/data/AnimalSpeak/animal_speak/train",
        "/cluster/work/boraa/data/AnimalSpeak/iNatural/train"
    ],
    "val": [
        "/cluster/work/boraa/data/AnimalSpeak/animal_speak/val",
        "/cluster/work/boraa/data/AnimalSpeak/iNatural/val"
    ]
}

audiocaps_path = {
    "train": "/cluster/work/boraa/data/AnimalSpeak/audiocaps/audiocaps/audio/train",
    "val": "/cluster/work/boraa/data/AnimalSpeak/audiocaps/audiocaps/audio/val"
}

# Initialize and load data
processor = CaptionDataProcessor(dataset_paths, audiocaps_path)
processor.load_data(force_reload=False)  # Set to False to load from saved files

NUM_OF_STEPS = 100_000
BATCH_SIZE = 32

for step in range(NUM_OF_STEPS):

# Build the train and validation pairs using the same function
train_pairs = processor.build_pairs(is_val=False)
val_pairs = processor.build_pairs(is_val=True)


# Create datasets
train_dataset = processor.create_dataset(*train_pairs)
val_dataset = processor.create_dataset(*val_pairs)


# Initialize the model trainer and train
trainer = ContrastiveModelTrainer("answerdotai/ModernBERT-base", train_dataset, val_dataset)
trainer.train()
