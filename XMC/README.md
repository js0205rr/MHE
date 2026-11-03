# MHE for XMC 

## Introduction
MHE-XMC makes no assumptions about the label space and does not use techniques such as HLT and label clustering for preprocessing. This suggests that additional, complex preprocessing and training tricks are not critical for the XMC task, and using simple label partitioning techniques are sufficient to process the XMC task.

## Preparation
The used datasets are download from 
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Amazon3M](https://drive.google.com/open?id=187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4) 

The pretrained model, including bert, roberta and xlnet, can be download from [Huggingface](https://huggingface.co).

## Quickly Start
When the dataset and the pretrained model are download, you can quickly run MHE-XMC by
```shell script
data_name = **
data_path = **
model_path = **
python src/main.py 
    --dataset $data_name 
    --data_path $data_path 
    --bert_path $model_path  
    --lr 1e-4 --epoch 20  
    --swa --swa_warmup 2
    --swa_step 100 
    --batch 16
    --num_group 172 
```
Note that when 'num_group' greater than 0, MHE-XMC use MHE for the XMC task. Otherwise, MHE-XMC is the simple multi-label classification method. See script 'run.sh' for detail setting.


## Training and Testing

### J1-A and K1-A backbone experiments / LF-AmazonTitles-131K

J1-A loads the **merged** `HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B` weights
and creates a fresh MHC LoRA adapter. It uses last-token pooling, selecting
the last position with `attention_mask=1` for each document. It does not load
the released `lora/` adapter a second time. K1-A loads the original
`KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` custom
`Qwen2Model` with its bidirectional attention and uses masked mean pooling.
The two model revisions checked on 2026-10-30 were
`e4a71aacd9b47ae6106fe57b7bf1c39992dc7ab1` (JevEmbed) and
`52c687bbe81a62a223c924698b787ec05c9a978a` (KaLM).

Both use the same lowercase product-title text and `This product text: `
prefix, 32-token truncation, 512 fixed contiguous label groups, top-10 group
routing, 300-dimensional label embeddings, group BCE plus candidate-label
BCE, BF16, batch 16, rank-64/alpha-128/dropout-0.1 LoRA, seed 6088, and
15 epochs. The backbone and head learning rates are `5e-5` and `1e-4`.
Validation is the fixed 4,000-example split (`random_state=1240`), and
checkpoint selection uses validation Label P@5. No test data is used to tune
the model. The released backbones' sentence-transformer normalization layers
are not used before the MHC heads.

Put complete model repositories in `$MODEL_ROOT` under their repository
basenames. Before training, run one real-batch compatibility check for each:

```bash
cd src
python smoke_backbones.py --experiment j1a --bert_path "$MODEL_ROOT" --data_path "$DATA_PATH"
python smoke_backbones.py --experiment k1a --bert_path "$MODEL_ROOT" --data_path "$DATA_PATH"
cd ..
```

Launch the full 15-epoch runs with distinct tags so existing results remain
untouched:

```bash
DATA_PATH=/path/to/data MODEL_ROOT=/path/to/pretrained RUN_TAG=j1a_v1 \
  bash run_lf_amazontitles_131k_jev_j1a.sh
DATA_PATH=/path/to/data MODEL_ROOT=/path/to/pretrained RUN_TAG=k1a_v1 \
  bash run_lf_amazontitles_131k_kalm_k1a.sh
```

J1-A versus the recorded Qwen **mean-pooling** baseline changes both weights
and pooling, so it does not isolate the effect of JevEmbed weights. The
2-epoch Qwen last-token pilot is not a mature 15-epoch control.

### Jina v3 base-only + fresh LoRA / LF-AmazonTitles-131K comparison

The Jina r64 run uses the native Transformers checkpoint
`jinaai/jina-embeddings-v3-hf` **without loading its pretrained rank-4 task
adapters**. It applies a fresh LoRA to Jina attention and feed-forward linear
layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc1`, `fc2`). Its LoRA settings
match the recorded Qwen r64 run (rank 64, alpha 128, dropout 0.1); the two
architectures do not have identical modules or trainable parameter counts.
The recorded Jina validation and test results therefore do not measure its
released `classification` task adapter. A clean comparison with that adapter
requires a new MHC training run from epoch 0 and a separate checkpoint.
The script keeps the same validation split, MHC heads, top-10 group routing,
batch size, sequence length, epoch count, seed, and learning rates. Product
titles remain lowercased. Qwen's `This product text:` prefix is not added to
Jina input. Neither backbone is L2-normalized before the MHC heads.

Install `requirements-jina-py312.txt` in a separate environment with a
CUDA-enabled PyTorch. Put the complete Jina HF checkpoint in
`$MODEL_ROOT/jina-embeddings-v3-hf` and the dataset in
`$DATA_PATH/LF-AmazonTitles-131K`. Before a full run, check one real batch:

```bash
cd src
python smoke_jina.py --bert_path "$MODEL_ROOT" --data_path "$DATA_PATH" --bf16
cd ..
```

The smoke check reads the same train/validation split and one real training
batch. It exercises tokenizer, LoRA and head gradients, group routing, and
label scores. A full training run requires a unique tag and refuses to
overwrite an existing checkpoint:

```bash
DATA_PATH=/path/to/data MODEL_ROOT=/path/to/pretrained RUN_TAG=jina_r64_seed6088_v1 \
  bash run_lf_amazontitles_131k_jina_h2_r64.sh
```

The script trains only after it is explicitly launched. Select checkpoints
on the 4000-item validation split; evaluate the official test separately.
Each validation epoch logs Label P@1/P@3/P@5, Group P@1/P@3/P@5,
`group_recall@10`, and `candidate_recall`. The two recall values average
per-document recall: the fraction of each document's true groups in the
top-10 routed groups, and the fraction of its true labels present in the
actual routed candidate list. Checkpoint selection still uses validation
Label P@5.

After selecting the best validation checkpoint, run the official test with
`src/test_jina.py`. It evaluates all 134,835 test documents, saves the
Top-10 routed label IDs and scores, and writes `metrics.json` with Label
P@1/P@3/P@5, Group P@1/P@3/P@5, Group Recall@10, Candidate Recall,
and SHA-256 hashes. The saved Top-10 ordering is independently checked
against the in-loop P@k calculation; requesting a separate Top-100 sort
can order tied scores differently.

### Original MHE training
Clone the code repository
```git
git clone git@github.com:Anoise/MHE.git
```

and go to the directory "MHE/XMC", run
```bash
bash run.sh [eurlex4k|wiki31k|amazon13k|amazon670k|wiki500k]
```

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.
- The hyperparameter "num_group" is the factorization of the total number of categories, which can be greater than the number of categories.
- The code partly refer to [LightXML](https://github.com/kongds/LightXML).

- Please refer to our [XMC-mGPUs](https://github.com/Anoise/MHE/tree/main/XMC-mGPUs) for multi-GPUs version.

## Performance

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC1.jpg">

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC2.jpg">
