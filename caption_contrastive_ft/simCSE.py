# simcse_animalspeak.py

import logging
import math
from datetime import datetime

from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses, LoggingHandler
from sentence_transformers.evaluation import TranslationEvaluator
from torch.utils.data import DataLoader

import pandas as pd

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# # Load AnimalSpeak from Hugging Face
# dataset = load_dataset("davidrrobinson/AnimalSpeak", split="train")
# dataset = dataset.filter(lambda x: x["species_common"] is not None and x["species_scientific"] is not None)
# # Convert to DataFrame
# df = dataset.to_pandas()

# # Drop duplicates first by common name
# df = df.drop_duplicates(subset="species_common")

# # Then drop duplicates by scientific name (now from the reduced df)
# df = df.drop_duplicates(subset="species_scientific")


df = pd.read_csv("species_names_filtered.csv")
# Back to HF Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.025, seed=42)

train_data = dataset["train"]
test_data = dataset["test"]

# Define model
model_name = 'sentence-transformers/all-mpnet-base-v2'
model_save_path = f'output/training_simcse-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
word_embedding_model = models.Transformer(model_name, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Prepare training samples
logging.info("Preparing training samples")
train_samples = [InputExample(texts=[row['en'], row['scientific_name']]) for row in train_data]


#human_labels= ["Clymene Dolphin", "Bottlenose Dolphin", "Spinner Dolphin", "Beluga, White Whale", "Bearded Seal", "Minke Whale", "Humpback Whale", "Southern Right Whale", "White-sided Dolphin", "Narwhal", "White-beaked Dolphin", "Northern Right Whale", "Frasers Dolphin", "Grampus, Rissos Dolphin", "Harp Seal", "Atlantic Spotted Dolphin", "Fin, Finback Whale", "Ross Seal", "Rough-Toothed Dolphin", "Killer Whale", "Pantropical Spotted Dolphin", "Short-Finned Pacific Pilot Whale", "Bowhead Whale", "False Killer Whale", "Melon Headed Whale", "Long-Finned Pilot Whale", "Striped Dolphin", "Leopard Seal", "Walrus", "Sperm Whale", "Common Dolphin"]
#scientific_labels = ['Stenella clymene', 'Tursiops', 'Stenella longirostris', 'Monodontidae', 'Erignathus', 'Balaenoptera acutorostrata', 'Megaptera', 'Eubalaena australis', 'Sagmatias obliquidens', 'Monodontidae', 'Lagenorhynchus albirostris', 'Lissodelphis borealis', 'Lagenodelphis hosei', 'Grampus', 'Pagophilus', 'Stenella frontalis', 'Balaenoptera physalus', 'Ommatophoca', 'Steno bredanensis', 'Orcinus orca', 'Stenella attenuata', 'Globicephala macrorhynchus', 'Balaenidae', 'Pseudorca crassidens', 'Peponocephala electra', 'Globicephala melas', 'Stenella', 'Hydrurga', 'Odobenidae', 'Physeteroidea', 'Tursiops truncatus']

#train_samples = [InputExample(texts=[x,y]) for x,y in zip(human_labels, scientific_labels)]



# Prepare evaluation
logging.info("Preparing dev/evaluation samples")
dev_evaluator = TranslationEvaluator(
    source_sentences=test_data["en"],
    target_sentences=test_data["scientific_name"],
    batch_size=64,
    name='animalspeak-dev'
)


# Configure training
train_batch_size = 128
num_epochs = 50
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesSymmetricRankingLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logging.info(f"Warmup steps: {warmup_steps}")

# Evaluate before training
logging.info("Performance before training")
dev_evaluator(model)



# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=100,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)

model.save_pretrained("models/ModernBERT-base/alligned3")