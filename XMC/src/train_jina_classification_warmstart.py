"""Five-epoch Jina classification-adapter warm start from an MHC r64 checkpoint.

The existing r64 adapter remains active and frozen. The released Jina
classification adapter (r4) and the existing MHC heads are optimized.
"""

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import get_balanced_groups, init_seed


MODEL_ID = 'jinaai/jina-embeddings-v3-hf'


def remap_official_adapter(adapter_dir, destination):
    """Map the released Jina adapter keys to the native HF model's module names."""
    original = load_file(str(adapter_dir / 'adapter_model.safetensors'), device='cpu')
    converted = {}
    for key, value in original.items():
        if key.startswith('roberta.encoder.layers.'):
            mapped = key.replace('roberta.encoder.layers.', 'layers.', 1)
        elif key.startswith('roberta.embeddings.'):
            mapped = key.replace('roberta.embeddings.', 'embeddings.', 1)
        elif key.startswith('roberta.pooler.'):
            mapped = key.replace('roberta.pooler.', 'pooler.', 1)
        else:
            raise RuntimeError(f'Unrecognized official adapter key: {key}')
        mapped = mapped.replace('.mixer.out_proj.', '.self_attn.o_proj.')
        if mapped in converted:
            raise RuntimeError(f'Duplicate mapped adapter key: {mapped}')
        converted[mapped] = value
    if len(converted) != 294:
        raise RuntimeError(f'Expected 294 official tensors, found {len(converted)}')
    shutil.copyfile(adapter_dir / 'adapter_config.json',
                    destination / 'adapter_config.json')
    save_file(converted, str(destination / 'adapter_model.safetensors'))
    return converted


def log_record(path, record):
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with path.open('a', encoding='utf-8') as output:
        output.write(line + '\n')


def validation_record(model, loader, epoch):
    group_p1, group_p3, group_p5, p1, p3, p5 = model.eval_epoch(epoch, loader)
    routing = model.last_routing_metrics
    return {
        'time': datetime.now().isoformat(timespec='seconds'),
        'epoch': epoch,
        'split': 'validation',
        'label_p1': p1,
        'label_p3': p3,
        'label_p5': p5,
        'group_p1': group_p1,
        'group_p3': group_p3,
        'group_p5': group_p5,
        'group_recall_at10': routing['group_recall_at10'],
        'candidate_recall': routing['candidate_recall'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    if args.epochs != 5:
        parser.error('This authorized run is fixed to five additional epochs')
    model_dir = args.model_root / 'jina-embeddings-v3-hf'
    adapter_dir = model_dir / 'classification'
    for required in (model_dir / 'model.safetensors',
                     adapter_dir / 'adapter_config.json',
                     adapter_dir / 'adapter_model.safetensors',
                     args.source_checkpoint):
        if not required.is_file():
            parser.error(f'Missing required input: {required}')
    if args.output_dir.exists():
        parser.error(f'Output directory already exists: {args.output_dir}')
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required')

    official = json.loads((adapter_dir / 'adapter_config.json').read_text(
        encoding='utf-8'))
    if official.get('r') != 4 or official.get('lora_alpha') != 1:
        raise RuntimeError('Unexpected official classification adapter rank/alpha')

    init_seed(6088)
    df, label_map = createDataCSV('lfamazontitles131k', path=str(args.data_path))
    _, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                   test_size=4000, random_state=1240)
    df.iloc[valid_df.index.values, 2] = 'valid'
    if (len(df[df.dataType == 'train']) != 290805
            or len(df[df.dataType == 'valid']) != 4000
            or len(label_map) != 131073):
        raise RuntimeError('Dataset or validation split differs from the source run')
    groups = get_balanced_groups(len(label_map), 512)

    model = XMLMHE(n_labels=len(label_map), group_y=groups,
                   bert=MODEL_ID, bert_path=str(args.model_root),
                   candidates_topk=10, hidden_dim=300,
                   use_lora=True, lora_r=64, lora_alpha=128,
                   lora_dropout=0.1, use_bf16=True, pooling='mean')
    source_state = torch.load(args.source_checkpoint, map_location='cpu',
                              weights_only=True)
    model.load_state_dict(source_state, strict=True)
    del source_state

    previous = model.bert.peft_config['default']
    if previous.r != 64 or previous.lora_alpha != 128:
        raise RuntimeError('The source checkpoint must use the old r64 adapter')
    with tempfile.TemporaryDirectory(prefix='jina_mhe_adapter_') as temporary:
        mapped_dir = Path(temporary)
        converted = remap_official_adapter(adapter_dir, mapped_dir)
        load_result = model.bert.load_adapter(
            str(mapped_dir), adapter_name='classification', is_trainable=True)
    unexpected = list(load_result.unexpected_keys)
    missing_classification = [name for name in load_result.missing_keys
                              if '.classification.' in name]
    if unexpected or missing_classification:
        raise RuntimeError(
            f'Official adapter load mismatch: unexpected={unexpected[:8]}, '
            f'missing classification={missing_classification[:8]}')
    named = dict(model.bert.named_parameters())
    for key, source in converted.items():
        if key.endswith('.weight'):
            parameter_key = 'base_model.model.' + key[:-7] + '.classification.weight'
        else:
            parameter_key = 'base_model.model.' + key + '.classification'
        loaded = named.get(parameter_key)
        if loaded is None or not torch.equal(loaded.detach().cpu(),
                                             source.to(dtype=loaded.dtype)):
            raise RuntimeError(f'Official adapter tensor was not loaded: {key}')
    model.bert.base_model.set_adapter(['default', 'classification'])

    for name, parameter in model.bert.named_parameters():
        if 'lora_' in name and ('.default.' in name or name.endswith('.default')):
            parameter.requires_grad_(False)

    model.cuda()

    trainable_backbone = [(name, parameter) for name, parameter
                          in model.named_parameters()
                          if name.startswith('bert.') and parameter.requires_grad]
    incorrect = [name for name, _ in trainable_backbone
                 if 'lora_' not in name or not (
                     '.classification.' in name or name.endswith('.classification'))]
    if not trainable_backbone or incorrect:
        raise RuntimeError(f'Backbone trainable parameters must be classification r4 only: {incorrect[:12]}')
    if any(('.default.' in name or name.endswith('.default')) and parameter.requires_grad
           for name, parameter in model.bert.named_parameters()):
        raise RuntimeError('Old r64 adapter is not frozen')
    heads = [(name, parameter) for name, parameter
             in model.named_parameters()
             if not name.startswith('bert.') and parameter.requires_grad]
    if not heads or not all(any(name.startswith(prefix) for name, _ in heads)
                            for prefix in ('l0.', 'l1.', 'embed.')):
        raise RuntimeError('MHC heads are not trainable')

    tokenizer = model.get_fast_tokenizer()
    train_set = MDataset(df, 'train', tokenizer, label_map, 32,
                         group_y=groups, candidates_num=3000,
                         model_name=MODEL_ID, dataset_name='lfamazontitles131k')
    valid_set = MDataset(df, 'valid', tokenizer, label_map, 32,
                         group_y=groups, candidates_num=3000,
                         model_name=MODEL_ID, dataset_name='lfamazontitles131k')
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              num_workers=5)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False,
                              num_workers=0)

    no_decay = ('bias', 'LayerNorm.weight')
    groups_for_optimizer = []
    for named, lr in ((trainable_backbone, 5e-5), (heads, 1e-4)):
        for decay in (True, False):
            parameters = [parameter for name, parameter in named
                          if (not any(term in name for term in no_decay)) == decay]
            if parameters:
                groups_for_optimizer.append({
                    'params': parameters,
                    'lr': lr,
                    'weight_decay': 0.01 if decay else 0.0,
                })
    optimizer = torch.optim.AdamW(groups_for_optimizer)
    expected = {id(parameter) for _, parameter in trainable_backbone + heads}
    actual = {id(parameter) for group in optimizer.param_groups
              for parameter in group['params']}
    if expected != actual:
        raise RuntimeError('Optimizer parameter set does not match trainable parameters')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / 'validation.jsonl'
    log_record(metrics_path, {
        'event': 'configuration', 'source_checkpoint': str(args.source_checkpoint),
        'model_dir': str(model_dir), 'official_adapter': str(adapter_dir),
        'active_adapters': ['default', 'classification'],
        'frozen_adapter': 'default/r64/alpha128',
        'trained_adapter': 'classification/r4/alpha1',
        'trainable_adapter_parameters': sum(p.numel() for _, p in trainable_backbone),
        'trainable_head_parameters': sum(p.numel() for _, p in heads),
        'adapter_load_unexpected_keys': len(unexpected),
        'adapter_load_missing_classification_keys': len(missing_classification),
        'verified_official_adapter_tensors': len(converted),
        'seed': 6088, 'epochs': 5, 'batch': 16, 'max_len': 32,
        'selection': 'validation label_p1',
    })

    best_p1 = -1.0
    for epoch in range(0, 6):
        if epoch:
            loss = model.train_epoch(epoch - 1, train_loader, optimizer,
                                     eval_loader=None)
        else:
            loss = None
        record = validation_record(model, valid_loader, epoch)
        record['train_loss'] = loss
        if record['label_p1'] > best_p1:
            best_p1 = record['label_p1']
            checkpoint_path = args.output_dir / f'checkpoint_epoch_{epoch}.bin'
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            model.save_model(str(checkpoint_path))
            record['selected_checkpoint'] = str(checkpoint_path)
        log_record(metrics_path, record)


if __name__ == '__main__':
    main()
