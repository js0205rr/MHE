"""One real MHC batch for J1-A or K1-A; no epoch or checkpoint is written."""

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import get_balanced_groups


EXPERIMENTS = {
    'j1a': ('HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B', 'last_token', 1024),
    'k1a': ('KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5', 'mean', 896),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--bert_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    model_name, pooling, hidden_size = EXPERIMENTS[args.experiment]
    model_dir = Path(args.bert_path) / model_name.rsplit('/', 1)[-1]
    if not (model_dir / 'model.safetensors').is_file():
        parser.error(f'Missing complete model checkpoint: {model_dir}')
    if not torch.cuda.is_available():
        parser.error('A CUDA device is required for the BF16 smoke check')

    torch.manual_seed(6088)
    df, label_map = createDataCSV('lfamazontitles131k', path=args.data_path)
    _, valid_df = train_test_split(df[df.dataType == 'train'], test_size=4000,
                                   random_state=1240)
    df.iloc[valid_df.index.values, 2] = 'valid'
    groups = get_balanced_groups(len(label_map), 512)
    model = XMLMHE(n_labels=len(label_map), group_y=groups, bert=model_name,
                   bert_path=args.bert_path, candidates_topk=10, hidden_dim=300,
                   use_lora=True, lora_r=64, lora_alpha=128, lora_dropout=0.1,
                   use_bf16=True, pooling=pooling).cuda()
    if model.bert.config.hidden_size != hidden_size:
        raise AssertionError('Unexpected backbone hidden size')
    tokenizer = model.get_tokenizer()
    dataset = MDataset(df, 'train', tokenizer, label_map, 32, group_y=groups,
                       candidates_num=3000, model_name=model_name,
                       dataset_name='lfamazontitles131k')
    batch = next(iter(DataLoader(dataset, batch_size=args.batch, num_workers=0)))
    inputs = [tensor.cuda() for tensor in batch[:5]]

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    lora = [p for name, p in trainable if name.startswith('bert.') and 'lora_' in name]
    if not lora or any('lora_' not in name for name, _ in trainable if name.startswith('bert.')):
        raise AssertionError('Backbone must train only a new MHC LoRA adapter')
    model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, loss = model(*inputs[:3], labels=inputs[3], group_labels=inputs[4])
    if not torch.isfinite(loss):
        raise AssertionError(f'Non-finite training loss: {loss.item()}')
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in lora):
        raise AssertionError('Missing or non-finite LoRA gradients')
    for head in (model.l0, model.l1, model.embed):
        if head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
            raise AssertionError('Missing or non-finite MHC head gradient')
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        group_logits, candidates, scores = model(*inputs[:3])
    if group_logits.shape != (args.batch, 512) or candidates.shape != scores.shape:
        raise AssertionError('Invalid routed prediction shapes')
    if not torch.isfinite(scores).all():
        raise AssertionError('Non-finite routed scores')
    group_hits, label_hits, count = model.get_routing_recall(
        group_logits, inputs[4], candidates, inputs[3], topk=10)
    print(f'{args.experiment} smoke OK; loss={loss.item():.4f}; '
          f'candidate_shape={tuple(candidates.shape)}; '
          f'group_recall@10={group_hits/count:.4f}; '
          f'candidate_recall={label_hits/count:.4f}; '
          f'peak_memory_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}')


if __name__ == '__main__':
    main()
