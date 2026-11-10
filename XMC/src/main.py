import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import MDataset, createDataCSV
from model import XMLMHE, is_embedding_backbone
from utils import *
import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--data_path', type=str, required=False, default='/home/user/Data')
parser.add_argument('--bert', type=str, required=False, default='bert-base')
parser.add_argument('--bert_path', type=str, required=False, default='../NLP-Model/')

# LoRA 相关参数
parser.add_argument('--use_lora', action='store_true', help='use LoRA for embedding backbone')
parser.add_argument('--lora_rank', type=int, default=256, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=256, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
parser.add_argument('--feature_dropout', type=float, default=0.5,
                    help='dropout on the pooled features shared by the group and label heads')
# 混合精度
parser.add_argument('--bf16', action='store_true', help='use bfloat16 mixed precision')

parser.add_argument('--max_len', type=int, required=False, default=512)
parser.add_argument('--pooling', choices=['mean', 'last_token'], default='mean')

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--num_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')
parser.add_argument('--run_tag', type=str, default='', help='unique suffix for logs, checkpoints, and predictions')
parser.add_argument('--no_save_checkpoint', action='store_true',
                    help='train and validate without saving any checkpoint')
parser.add_argument('--selection_metric', choices=['p1', 'p5'], default='p1',
                    help='validation label precision used to select the saved checkpoint')

args = parser.parse_args()
if not 0.0 <= args.feature_dropout < 1.0:
    parser.error('--feature_dropout must be in [0, 1)')


def train(model, df, label_map):
    
    tokenizer = model.get_tokenizer()

    print('dataset = ', args.dataset)
    train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                        candidates_num=args.group_y_candidate_num, model_name=args.bert,
                        dataset_name=args.dataset)
    test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, group_y=group_y,
                        candidates_num=args.group_y_candidate_num, model_name=args.bert,
                        dataset_name=args.dataset)

    train_d.tokenizer = model.get_fast_tokenizer()
    test_d.tokenizer = model.get_fast_tokenizer()

    trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                shuffle=True)
    testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                            shuffle=False)
    if args.valid:
        print('valid ...')
        valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                            candidates_num=args.group_y_candidate_num, model_name=args.bert,
                            dataset_name=args.dataset)
        validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0,
                                    shuffle=False)
   
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']

# For embedding backbones, use different learning rates for backbone and classification head
    # Backbone (bert): args.lr, Classification head (l0, l1, embed): args.lr * 2
    if is_embedding_backbone(args.bert):
        backbone_params = []
        backbone_params_no_decay = []
        head_params = []
        head_params_no_decay = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bert' in n:
                # Backbone parameters (including LoRA)
                if any(nd in n for nd in no_decay):
                    backbone_params_no_decay.append(p)
                else:
                    backbone_params.append(p)
            else:
                # Classification head parameters (l0, l1, embed, etc.)
                if any(nd in n for nd in no_decay):
                    head_params_no_decay.append(p)
                else:
                    head_params.append(p)

        optimizer_grouped_parameters = [
            {'params': backbone_params, 'weight_decay': 0.01, 'lr': args.lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': args.lr},
            {'params': head_params, 'weight_decay': 0.01, 'lr': args.lr * 2},
            {'params': head_params_no_decay, 'weight_decay': 0.0, 'lr': args.lr * 2},
        ]
        print(f'Using differential learning rates: backbone={args.lr}, head={args.lr * 2}')
        print(f'Trainable parameters: backbone={sum(p.numel() for p in backbone_params + backbone_params_no_decay)}, '
              f'head={sum(p.numel() for p in head_params + head_params_no_decay)}')
    else:
        # For other models (BERT, RoBERTa, XLNet), use unified learning rate
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print(f'Using unified learning rate: {args.lr}')

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)


    best_selection_score = -1
    LOG.log(f'configuration: backbone={args.bert}, source={model.model_source}, '
            f'pooling={args.pooling}, seed={args.seed}, batch={args.batch}, '
            f'max_len={args.max_len}, epochs={args.epoch}, lr={args.lr}, '
            f'lora={args.use_lora}, rank={args.lora_rank}, alpha={args.lora_alpha}, '
            f'lora_dropout={args.lora_dropout}, feature_dropout={args.feature_dropout}, '
            f'head_lr={args.lr * 2 if is_embedding_backbone(args.bert) else args.lr}, '
            f'bf16={args.bf16}, swa={args.swa}, '
            f'save_checkpoint={not args.no_save_checkpoint}, '
            f'selection_metric=label_{args.selection_metric}, '
            f'groups={args.num_group}, routed_groups={args.group_y_candidate_topk}')
    for epoch in range(0, args.epoch): ##
        train_loss = model.train_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        if args.valid:
            ev_result = model.eval_epoch(epoch, validloader)
        else:
            ev_result = model.eval_epoch(epoch, testloader)

        g_p1, g_p3, g_p5, p1, p3, p5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
        if args.num_group>0:
            components = model.last_train_losses
            log_str += (f' group_bce:{components["group_bce"]:.6f}'
                        f' label_bce:{components["label_bce"]:.6f}')
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
            routing = model.last_routing_metrics
            log_str += (f' group_recall@10:{routing["group_recall_at10"]:.4f}'
                        f' candidate_recall:{routing["candidate_recall"]:.4f}')
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        selection_score = p1 if args.selection_metric == 'p1' else p5
        if not args.no_save_checkpoint and best_selection_score < selection_score:
            best_selection_score = selection_score
            checkpoint_path = f'models/model-{exp_name}.bin'
            temp_checkpoint_path = checkpoint_path + '.tmp'
            model.save_model(temp_checkpoint_path)
            os.replace(temp_checkpoint_path, checkpoint_path)
            metadata_path = checkpoint_path + '.meta.json'
            with open(metadata_path + '.tmp', 'w', encoding='utf-8') as metadata_file:
                json.dump({'arguments': vars(args), 'model_source': model.model_source,
                           'selected_epoch': epoch,
                           'selection_metric': f'{"validation" if args.valid else "test"}_label_{args.selection_metric}',
                           'selection_value': selection_score,
                           'validation_label_p1': p1 if args.valid else None,
                           'validation_label_p3': p3 if args.valid else None,
                           'validation_label_p5': p5 if args.valid else None},
                          metadata_file, indent=2, sort_keys=True)
            os.replace(metadata_path + '.tmp', metadata_path)
            LOG.log(f'checkpoint selected: epoch={epoch}, label_{args.selection_metric}={selection_score:.4f}, path={checkpoint_path}')

if __name__ == '__main__':
    init_seed(args.seed)
    exp_name = get_exp_name(args.dataset,args.bert,args.num_group)
    if args.run_tag:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_tag):
            raise ValueError('--run_tag may contain only letters, digits, underscores, and hyphens')
        exp_name += '_' + args.run_tag
    checkpoint_path = f'models/model-{exp_name}.bin'
    if not args.eval_model and os.path.exists(checkpoint_path):
        raise FileExistsError(f'Checkpoint already exists: {checkpoint_path}. Choose a new --run_tag.')
    LOG = Logger('log_'+exp_name)
    
    print(f'load {args.dataset} dataset...')
    df, label_map = createDataCSV(args.dataset,path=args.data_path)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    if args.num_group>0:
        if args.dataset == 'lfamazontitles131k':
            group_y = get_balanced_groups(len(label_map), args.num_group)
        else:
            num_classes, per_ele_classes = check_gorup(len(label_map), args.num_group)
            group_y = get_groups_v2(len(label_map), num_classes, per_ele_classes)
    else:
        group_y = None
    model = XMLMHE(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                        bert_path=args.bert_path,update_count=args.update_count,
                        use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                        candidates_topk=args.group_y_candidate_topk,
                        hidden_dim=args.hidden_dim,
                        use_lora=args.use_lora,
                        lora_r=args.lora_rank,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        dropout=args.feature_dropout,
                        use_bf16=args.bf16,
                        pooling=args.pooling)
    
    if args.eval_model and args.dataset in ['wiki500k', 'amazon670k', 'amazon3m', 'lfamazontitles131k']:
        print(f'load models/model-{exp_name}.bin ......')
        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num, model_name=args.bert,
                           dataset_name=args.dataset)

        test_d.tokenizer = model.get_fast_tokenizer()

        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)

        model.load_state_dict(torch.load(f'models/model-{exp_name}.bin',map_location='cpu'))
        model = model.cuda()

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{exp_name}-labels.npy', np.array(pred_labels))
        np.save(f'results/{exp_name}-scores.npy', np.array(pred_scores))
    else:
        train(model, df, label_map)
