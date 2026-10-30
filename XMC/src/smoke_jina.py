"""One real-data Jina/MHC batch; does not train an epoch or save a checkpoint."""

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import get_balanced_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', required=True,
                        help='parent directory of jina-embeddings-v3-hf')
    parser.add_argument('--data_path', required=True,
                        help='parent directory of LF-AmazonTitles-131K')
    parser.add_argument('--bf16', action='store_true')
    args = parser.parse_args()
    checkpoint_dir = Path(args.bert_path) / 'jina-embeddings-v3-hf'
    if not (checkpoint_dir / 'config.json').is_file():
        parser.error(f'Missing local Jina checkpoint: {checkpoint_dir}')
    if args.bf16 and not torch.cuda.is_available():
        parser.error('--bf16 requires a CUDA device')

    torch.manual_seed(6088)
    df, label_map = createDataCSV('lfamazontitles131k', path=args.data_path)
    _, valid_df = train_test_split(df[df['dataType'] == 'train'], test_size=4000,
                                   random_state=1240)
    df.iloc[valid_df.index.values, 2] = 'valid'
    groups = get_balanced_groups(len(label_map), 512)
    model = XMLMHE(
        n_labels=len(label_map),
        group_y=groups,
        bert='jinaai/jina-embeddings-v3-hf',
        bert_path=args.bert_path,
        candidates_topk=10,
        hidden_dim=300,
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        use_bf16=args.bf16,
    )
    tokenizer = model.get_tokenizer()
    train_dataset = MDataset(
        df, 'train', tokenizer, label_map, 32, group_y=groups,
        candidates_num=3000, model_name='jinaai/jina-embeddings-v3-hf',
        dataset_name='lfamazontitles131k',
    )
    batch = next(iter(DataLoader(train_dataset, batch_size=2, num_workers=0)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if tokenizer.pad_token_id != 1:
        raise AssertionError(f'Unexpected Jina pad token: {tokenizer.pad_token_id}')
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_ids = batch[2].to(device)
    labels = batch[3].to(device)
    group_labels = batch[4].to(device)

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    lora = [(name, p) for name, p in trainable if 'lora_' in name]
    if not lora or any('lora_' not in name for name, _ in trainable if name.startswith('bert.')):
        raise AssertionError('Expected only new LoRA parameters to train in Jina backbone')
    print(f'Jina trainable LoRA parameters: {sum(p.numel() for _, p in lora)}')

    model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.bf16 and device == 'cuda'):
        _, loss = model(input_ids, attention_mask, token_type_ids,
                        labels=labels, group_labels=group_labels)
    if not torch.isfinite(loss):
        raise AssertionError(f'Non-finite training loss: {loss}')
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for _, p in lora):
        raise AssertionError('No finite Jina LoRA gradient')
    if model.l0.weight.grad is None or model.l1.weight.grad is None or model.embed.weight.grad is None:
        raise AssertionError('Missing MHC head gradient')
    torch.optim.AdamW([p for _, p in trainable], lr=5e-5).step()

    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                         enabled=args.bf16 and device == 'cuda'):
        group_logits, candidates, scores = model(input_ids, attention_mask, token_type_ids)
    if group_logits.shape != (2, 512) or candidates.shape != scores.shape:
        raise AssertionError('Invalid routed inference output shapes')
    if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise AssertionError('Invalid routed inference scores')
    if candidates.min() < 0 or candidates.max() >= len(label_map):
        raise AssertionError('Candidate label out of range')
    group_recall, candidate_recall, count = model.get_routing_recall(
        group_logits, group_labels, candidates, labels, topk=10)
    if count != 2 or not (0 <= group_recall <= count and 0 <= candidate_recall <= count):
        raise AssertionError('Invalid routed recall metrics')
    print(f'Jina/MHC train and routed inference OK; loss={loss.item():.4f}; '
          f'candidate_shape={tuple(candidates.shape)}; '
          f'group_recall@10={group_recall / count:.4f}; '
          f'candidate_recall={candidate_recall / count:.4f}')


if __name__ == '__main__':
    main()
