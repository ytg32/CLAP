# CLAP
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/logo.PNG" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2211.06687"><img src="https://img.shields.io/badge/arXiv-2211.06687-brightgreen.svg?style=flat-square"/></a>
  <a href="https://pypi.org/project/laion-clap"><img src="https://badge.fury.io/py/laion-clap.svg"/></a>
  <a href="https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/clap"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue"/></a>
</p>

## AnimalSpeak Thesis Code (CLAP-based)

This repository provides open-source code and resources for my thesis, which utilizes **Contrastive Language-Audio Pretraining (CLAP)** to learn from audio and language pairs.  
It is based on [LAION-AI's CLAP](https://github.com/LAION-AI/CLAP).

---


## Environment Installation
If you want to check and reuse our model into your project instead of directly using the pip library, you need to install the same environment as we use, please run the following command:
bash
conda create env -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
# you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

## Dataset format
We use training data in webdataset format. For details of our dataset please see https://github.com/LAION-AI/audio-dataset.

Due to copyright reasons, we cannot release the dataset we train this model on. However, we released [LAION-audio-630K](https://github.com/LAION-AI/audio-dataset/tree/main/laion-audio-630k), the data source we used to compose the dataset with link to each audio and their caption. Please refer to [LAION-audio-630K](https://github.com/LAION-AI/audio-dataset/tree/main/laion-audio-630k) for more details. You could download the dataset, preprocess it on your own and train it locally. To train on the local dataset, please change the --remotedata in training scripts (see [experiment_scripts](./experiment_scripts) folder) with --datasetpath <your dir to datasets>.

You can find an example of our dataset format in [here](https://drive.google.com/drive/folders/1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX?usp=sharing).
It contains the full ESC50 dataset, split according to the first 5-fold split.



## Datasets & Preprocessing

### Datasets Used:
- **AudioCaps**  
- **iNatSounds (iNatural)**  
- **Watkins (marine mammals)**

All datasets are converted into **WebDataset** format before training.

---

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
The scripts included there are the one we used to train our model on a SLURM cluster. 
You need to change the script to fit your own environment.
you can run sbatch [experiment_scripts](./experiment_scripts)
We use [Weights and Biases](https://wandb.ai/site) for experiment logging. You need to configure the weights and biases in your environment.
To train on local dataset, please change the --remotedata in training scripts (see [experiment_scripts](./experiment_scripts) folder) with --datasetpath <your dir to datasets>.

## Core Code
Please refer to [main.py](https://github.com/LAION-AI/CLAP/blob/laion_clap_pip/src/laion_clap/training/main.py), [train.py](https://github.com/LAION-AI/CLAP/blob/laion_clap_pip/src/laion_clap/training/train.py), [data.py](https://github.com/LAION-AI/CLAP/blob/laion_clap_pip/src/laion_clap/training/data.py),and [model.py](https://github.com/LAION-AI/CLAP/blob/laion_clap_pip/src/laion_clap/clap_module/model.py) to quicly get familiar with our model.

## Citation
If you find this project and the LAION-Audio-630K dataset useful, please cite our paper:

@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}
@inproceedings{htsatke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2022}
}


## Acknowledgements

This project is working in progress, thus the codebase and model might not be perfect or bug-free. 
We will very much appreciate any kind of contribution or and issue raised.
If you find a bug or have any suggestion, please feel free to open an issue or contact us.
If you would actively contribute to this project, please join the discord of LAION.