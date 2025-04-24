## AnimalSpeak Thesis Code (CLAP-based)

This repository provides open-source code and resources for my thesis, which utilizes **Contrastive Language-Audio Pretraining (CLAP)** to learn from audio and language pairs.  
It is based on [LAION-AI's CLAP](https://github.com/LAION-AI/CLAP).

---


## Environment Installation

```bash
conda create env -n clap python=3.9
conda activate clap
git clone https://github.com/ytg32/CLAP.git
cd CLAP
pip install -r requirements.txt
  ```

---

## Dataset format
We use training data in webdataset format. You can find an example of our dataset format in [here](https://drive.google.com/drive/folders/1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX?usp=sharing).
It contains the full ESC50 dataset, split according to the first 5-fold split.


## Datasets & Preprocessing

### Datasets Used:
- **AudioCaps**  
- **iNatSounds (iNatural)**  
- **Watkins (marine mammals)**

All datasets are converted into **WebDataset** format before training.

### Data Preparation Steps

#### 1. AudioCaps

- The data can be downloaded from [this link](https://www.kaggle.com/datasets/nickkar30/audiocaps).

To generate `.json` metadata:

```bash
python CLAP/data/helpers/audiocaps_json_generator.py \
  --tsv_path /path/to/{train,audiocaps_val_new,audiocaps_test_new}.tsv \
  --output_dir /path/to/data/audiocaps/{train,val,test}
  ```

#### 2. iNatSounds (iNatural)

- The data can be downloaded from [this link](https://github.com/visipedia/inat_sounds/tree/main/2024). Only the recordings are needed; there is no need to download annotations.

To generate `.json` metadata:

```bash
python CLAP/data/helpers/python create_taxonomy_json.py \
  --data_path /path/to/data/iNatural/{train, val, test}
```

#### 3. Watkins

The data can be downloaded from [this link]()

### Webdataset Generation

Before generating the WebDataset, the data should be structured like this:

    data
    └─ audiocaps
       └─ train
            └─ 0.wav
            └─ 1.wav
            └─ 2.wav
            └─ 3.wav
       └─ val
       └─ test
    └─ iNatural
       └─ train
       └─ val
       └─ test
    └─ watkins
       └─ train
       └─ val
       └─ test

WebDataset can be generated using:

```bash
python CLAP/data/helpers/shard_dataset.py \
--input_dir /path/to/data/{audiocaps, watkins, iNatural}/{train, val, test} \
--output_dir /path/to/webdataset/{audiocaps, watkins, iNatural}/{train, val, test} \
--samples_per_shard 1024
```

Before running the script, the following checks should be made:

* All sound files should have a sampling rate of 16,000 Hz and the ".wav" extension.

* Every JSON and WAV file should have its corresponding pair. To check if every WAV and JSON file has a pair, you can use:

```bash
python CLAP/data/helpers/check_pairs.py \
    /path/to/data/{audiocaps, watkins, iNatural}/{train, val, test}
```

After running the code, the folder structure of the WebDataset should look like this:

    webdataset
    └─ audiocaps
       └─ train
            └─ 00000.tar
            └─ 00001.tar
            └─ 00002.tar
            └─ sizes.json
       └─ val
       └─ test
    └─ iNatural
       └─ train
       └─ val
       └─ test
    └─ watkins
       └─ train
       └─ val
       └─ test



## Training, Fine-tuning and Evaluation
Please find the script of training, in the [experiment_scripts](./experiment_scripts) folder. 
The scripts included there are the one used to train the model on a SLURM cluster. 

