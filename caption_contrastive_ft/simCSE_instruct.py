from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    LoggingHandler
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TranslationEvaluator

import logging
import math
from datetime import datetime


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)




# Load AnimalSpeak from Hugging Face
dataset = load_dataset("davidrrobinson/AnimalSpeak", split="train")
dataset = dataset.filter(lambda x: x["species_common"] is not None and x["species_scientific"] is not None)
# Convert to DataFrame
df = dataset.to_pandas()[["species_common", "species_scientific"]]

# Drop duplicates first by common name
df = df.drop_duplicates(subset="species_common")

# Then drop duplicates by scientific name (now from the reduced df)
df = df.drop_duplicates(subset="species_scientific")

# Back to HF Dataset
dataset = Dataset.from_pandas(df, preserve_index=False)
dataset = dataset.train_test_split(test_size=0.025, seed=42)

train_data = dataset["train"]
test_data = dataset["test"]

# Configure training
num_epochs = 1
train_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/Qwen-instruct",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed,
    prompts= "Instruct: Given either the scientific or common name of an animal, retrieve the corresponding name in the scientific format.\nQuery: ",

)

logging.info("Preparing dev/evaluation samples")
dev_evaluator = TranslationEvaluator(
    source_sentences=test_data["species_common"],
    target_sentences=test_data["species_scientific"],
    batch_size=8,
    name='animalspeak-dev'
)


# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Save the trained model
model.save_pretrained("models/qwen/final")