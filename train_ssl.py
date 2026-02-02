import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MultiModalResNet1D, BasicBlock
from utils import read_conf, EEGDataset, ModelMoCo


class MultiEpochDataLoader:
    def __init__(self, data_loader, num_epochs):
        self.data_loader = data_loader
        self.num_epochs = num_epochs

    def __iter__(self):
        for _ in range(self.num_epochs):
            for batch in self.data_loader:
                yield batch

    def __len__(self):
        return self.num_epochs * len(self.data_loader)


def train_epoch(net, data_loader, optimizer, epoch, args):
    net.train()
    total_loss, total_num = 0.0, 0
    
    for im_1, im_2, _, _ in tqdm(data_loader, desc=f'Epoch {epoch}'):
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        
        loss = net(im_1, im_2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += im_1.shape[0]
        total_loss += loss.item() * im_1.shape[0]
    
    if total_num == 0:
        print("Warning: No data processed in this epoch!")
        return 0.0
    
    return total_loss / total_num


def adjust_learning_rate(optimizer, epoch, args, eta_min=1e-5):
    lr = args.lr
    if args.cos:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        lr = eta_min + (lr - eta_min) * cosine_decay
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='Self-supervised Training with MoCo')
    parser.add_argument('--cfg', type=str, default='configs/ssl_example.cfg', help='Path to config file')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--moco-dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--moco-k', type=int, default=4096, help='Queue size')
    parser.add_argument('--moco-m', type=float, default=0.99, help='Momentum')
    parser.add_argument('--moco-t', type=float, default=0.08, help='Temperature')
    parser.add_argument('--cos', action='store_true', help='Use cosine LR schedule')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='LR schedule')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--results-dir', type=str, default='', help='Output directory')
    parser.add_argument('--fold-idx', type=int, default=15, help='Fold index used when list files are not explicit')
    parser.add_argument('--multi-epoch-repeat', type=int, default=30, help='Repeat dataloader iterations per epoch')
    
    args = parser.parse_args()
    
    options = read_conf(args.cfg)
    
    channel_lis = options.channel_num.split(':')
    if len(channel_lis) == 1:
        channel_num1 = 0
        channel_num2 = int(channel_lis[0])
        channel_num = channel_num2 - channel_num1
        channel_lis = None
    elif len(channel_lis) == 2:
        channel_num1 = int(channel_lis[0])
        channel_num2 = int(channel_lis[1])
        channel_num = channel_num2 - channel_num1
        channel_lis = None
    else:
        channel_num = len(channel_lis)
        channel_num1 = None
        channel_num2 = None
        channel_lis = [int(x) for x in channel_lis]

    options.wlen = int(int(options.fs) * int(options.cw_len) / 1000.0)
    
    arch = options.arch
    cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))

    if arch == 'MultiModalResNet1D':
        option_my = {
            'modalities': {
                'eeg': {
                    'channel_num': channel_num,
                    'cnn_N_filt': cnn_N_filt,
                    'cnn_len_filt': cnn_len_filt,
                    'fs': int(options.fs),
                    'input_dim': options.wlen,
                    'block': 'BasicBlock',
                    'structure': 'resnet18'
                },
            }
        }
        block_types = {'BasicBlock': BasicBlock}
        layers_dict = {'resnet18': [4, 4, 4, 4]}
        encoder_q = MultiModalResNet1D(option_my, block_types, layers_dict,
                                       num_classes=args.moco_dim, token_len=32, token_dim=32, attn_heads=4)
        encoder_k = MultiModalResNet1D(option_my, block_types, layers_dict,
                                       num_classes=args.moco_dim, token_len=32, token_dim=32, attn_heads=4)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Only 'MultiModalResNet1D' is supported.")
    
    model = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t,
                     encoder_q=encoder_q, encoder_k=encoder_k).cuda()
    
    if args.results_dir == '':
        args.results_dir = f'output/ssl_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(args.results_dir, exist_ok=True)
    
    tr_lst = options.tr_lst
    if not tr_lst.endswith('.txt'):
        tr_lst = tr_lst + f'{args.fold_idx}.txt'
    
    train_data = EEGDataset(
        tr_lst,
        options.data_folder + '/',
        options.name,
        options.wlen,
        channel_num,
        channel_num1,
        channel_num2,
        channel_lis,
        emotion_state=getattr(options, 'emotion_state', 'valence')
    )

    drop_last = len(train_data) >= args.batch_size
    train_loader_one = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last
    )
    train_loader = MultiEpochDataLoader(train_loader_one, num_epochs=args.multi_epoch_repeat)

    print(f'Dataset size: {len(train_data)}, Batch size: {args.batch_size}, Drop last: {drop_last}, Repeat: {args.multi_epoch_repeat}')
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    
    epoch_start = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
    
    for epoch in range(epoch_start, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(model, train_loader, optimizer, epoch, args)
        print(f'Epoch {epoch}/{args.epochs}, Loss: {train_loss:.4f}')
        
        if epoch % 10 == 0:
            checkpoint = {
                'CNN_model_par': model.encoder_q.state_dict(),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.results_dir, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()

