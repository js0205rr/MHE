"""Evaluate a trained Jina/MHE checkpoint on the official test split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import get_balanced_groups


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_root', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='evaluate 16 test rows without writing files')
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if not args.smoke and args.output_dir.exists():
        parser.error(f'Output directory already exists: {args.output_dir}')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('CUDA with BF16 support is required')

    torch.manual_seed(6088)
    df, label_map = createDataCSV('lfamazontitles131k', path=args.data_path)
    groups = get_balanced_groups(len(label_map), 512)
    model = XMLMHE(
        n_labels=len(label_map), group_y=groups,
        bert='jinaai/jina-embeddings-v3-hf', bert_path=args.model_root,
        candidates_topk=10, hidden_dim=300,
        use_lora=True, lora_r=64, lora_alpha=128, lora_dropout=0.1,
        use_bf16=True,
    )
    test_dataset = MDataset(
        df, 'test', model.get_tokenizer(), label_map, 32,
        group_y=groups, candidates_num=3000,
        model_name='jinaai/jina-embeddings-v3-hf',
        dataset_name='lfamazontitles131k',
    )
    test_dataset.tokenizer = model.get_fast_tokenizer()
    if len(test_dataset) != 134835:
        raise AssertionError(f'Unexpected official test size: {len(test_dataset)}')
    eval_dataset = Subset(test_dataset, range(16)) if args.smoke else test_dataset
    loader = DataLoader(eval_dataset, batch_size=16,
                        num_workers=0 if args.smoke else 5, shuffle=False)

    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.cuda()
    scores, predicted_labels = model.eval_epoch(
        0, loader, mode='test', prediction_topk=10)
    metrics = model.last_eval_metrics
    if scores.shape != (len(eval_dataset), 10) or predicted_labels.shape != scores.shape:
        raise AssertionError('Unexpected test prediction shapes')
    if not np.isfinite(scores).all() or predicted_labels.min() < 0 or predicted_labels.max() >= len(label_map):
        raise AssertionError('Invalid test predictions')

    # Independent check: the saved top-10 predictions must reproduce the
    # in-loop Label P@k calculation from their first k labels.
    matches = np.zeros(3, dtype=np.int64)
    for predicted, truth in zip(predicted_labels, test_dataset.df.label.values[:len(eval_dataset)]):
        true_labels = {label_map[label] for label in truth.split()}
        matches += [len(set(predicted[:k]) & true_labels) for k in (1, 3, 5)]
    reproduced = matches / len(eval_dataset) / np.array([1, 3, 5])
    if not np.allclose(reproduced, [metrics['label_p1'], metrics['label_p3'], metrics['label_p5']], atol=1e-8):
        raise AssertionError(f'Saved predictions disagree with Label P@k: {reproduced} vs {metrics}')

    metrics.update({
        'split': 'official_test',
        'seed': 6088,
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256(args.checkpoint),
        'model': 'jinaai/jina-embeddings-v3-hf',
        'epochs_trained': 15,
        'validation_selected_epoch': 14,
    })
    if args.smoke:
        print('SMOKE TEST:', json.dumps(metrics, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=False)
    labels_path = args.output_dir / 'labels.npy'
    scores_path = args.output_dir / 'scores.npy'
    np.save(labels_path, predicted_labels)
    np.save(scores_path, scores)
    metrics['labels_sha256'] = sha256(labels_path)
    metrics['scores_sha256'] = sha256(scores_path)
    (args.output_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('OFFICIAL TEST:', json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
