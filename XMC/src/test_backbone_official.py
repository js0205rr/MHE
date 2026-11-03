"""Evaluate a validation-selected JevEmbed or KaLM MHC checkpoint on official Test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import get_balanced_groups


EXPERIMENTS = {
    'j1a': {
        'backbone': 'HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B',
        'pooling': 'last_token',
        'run_tag': 'j1a_last_r64_seed6088_20261030',
    },
    'k1a': {
        'backbone': 'KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5',
        'pooling': 'mean',
        'run_tag': 'k1a_mean_r64_seed6088_20261030',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_root', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if args.output_dir.exists():
        parser.error(f'Output directory already exists: {args.output_dir}')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('CUDA with BF16 support is required')

    config = EXPERIMENTS[args.experiment]
    torch.manual_seed(6088)
    df, label_map = createDataCSV('lfamazontitles131k', path=args.data_path)
    if len(label_map) != 131073:
        raise AssertionError(f'Unexpected label count: {len(label_map)}')
    groups = get_balanced_groups(len(label_map), 512)
    model = XMLMHE(
        n_labels=len(label_map), group_y=groups,
        bert=config['backbone'], bert_path=args.model_root,
        candidates_topk=10, hidden_dim=300,
        use_lora=True, lora_r=64, lora_alpha=128, lora_dropout=0.1,
        use_bf16=True, pooling=config['pooling'],
    )
    test_dataset = MDataset(
        df, 'test', model.get_tokenizer(), label_map, 32,
        group_y=groups, candidates_num=3000,
        model_name=config['backbone'], dataset_name='lfamazontitles131k',
    )
    test_dataset.tokenizer = model.get_fast_tokenizer()
    if len(test_dataset) != 134835:
        raise AssertionError(f'Unexpected official Test size: {len(test_dataset)}')
    loader = DataLoader(test_dataset, batch_size=16, num_workers=5, shuffle=False)

    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.cuda()
    scores, predicted_labels = model.eval_epoch(0, loader, mode='test', prediction_topk=10)
    metrics = model.last_eval_metrics
    expected_shape = (len(test_dataset), 10)
    if scores.shape != expected_shape or predicted_labels.shape != expected_shape:
        raise AssertionError(f'Unexpected prediction shapes: {scores.shape}, {predicted_labels.shape}')
    if not np.isfinite(scores).all() or predicted_labels.min() < 0 or predicted_labels.max() >= len(label_map):
        raise AssertionError('Invalid Test predictions')

    # The saved Top-10 predictions must reproduce the in-loop P@k values.
    matches = np.zeros(3, dtype=np.int64)
    for predicted, truth in zip(predicted_labels, test_dataset.df.label.values):
        true_labels = {label_map[label] for label in truth.split() if label in label_map}
        matches += [len(set(predicted[:k]) & true_labels) for k in (1, 3, 5)]
    reproduced = matches / len(test_dataset) / np.array([1, 3, 5])
    reported = [metrics['label_p1'], metrics['label_p3'], metrics['label_p5']]
    if not np.allclose(reproduced, reported, atol=1e-8):
        raise AssertionError(f'Saved predictions disagree with Label P@k: {reproduced} vs {reported}')

    metrics.update({
        'split': 'official_test',
        'dataset': 'LF-AmazonTitles-131K',
        'experiment': args.experiment,
        'backbone': config['backbone'],
        'pooling': config['pooling'],
        'run_tag': config['run_tag'],
        'seed': 6088,
        'epochs_trained': 15,
        'validation_selected_epoch_zero_based': 14,
        'validation_selection_metric': 'label_p5',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256(args.checkpoint),
        'prediction_topk': 10,
        'label_precision_percent': {
            'P@1': metrics['label_p1'] * 100,
            'P@3': metrics['label_p3'] * 100,
            'P@5': metrics['label_p5'] * 100,
        },
    })
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(exist_ok=False)
    labels_path = args.output_dir / 'labels.npy'
    scores_path = args.output_dir / 'scores.npy'
    np.save(labels_path, predicted_labels)
    np.save(scores_path, scores)
    metrics['labels_sha256'] = sha256(labels_path)
    metrics['scores_sha256'] = sha256(scores_path)
    (args.output_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('OFFICIAL TEST:', json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
