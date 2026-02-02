import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MultiModalResNet1D, BasicBlock, MoEEmotionClassifier
from utils import ReadList, read_conf


def create_batches(batch_size, data_folder, wav_lst, N_snt, wlen, channel_num,
                   channel_num1, channel_num2, channel_lis, dataset_name, emotion_state, num_label_dims=1):
    sig_batch = np.zeros([batch_size, channel_num, wlen])
    if num_label_dims > 1:
        lab_batch = np.zeros((batch_size, num_label_dims))
    else:
        lab_batch = np.zeros(batch_size)
    
    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    
    for i in range(batch_size):
        wav_file = wav_lst[snt_id_arr[i]]
        file_path = os.path.join(data_folder, wav_file)
        
        if not os.path.exists(file_path):
            parts = wav_file.split('_')
            if len(parts) >= 5 and parts[2] in ['0', '1', 'mid']:
                fixed_parts = parts[:2] + parts[3:]
                fixed_file = '_'.join(fixed_parts)
                file_path = os.path.join(data_folder, fixed_file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file not found: {wav_file} and {fixed_file}")
                wav_lst[snt_id_arr[i]] = fixed_file
        
        data = scipy.io.loadmat(file_path)
        
        if dataset_name == 'seed':
            signal = data['eeg']
        else:
            signal = data['data']
        
        signal = signal[:, :]
        if channel_lis is None or len(channel_lis) <= 2:
            signal = signal[channel_num1:channel_num2, :]
        else:
            signal = signal[channel_lis, :]
        
        snt_len = signal.shape[-1]
        if snt_len < wlen:
            pad_width = wlen - snt_len
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
            snt_len = wlen
        
        snt_beg = np.random.randint(snt_len - wlen + 1)
        snt_end = snt_beg + wlen
        sig_batch[i, :, :] = signal[:, snt_beg:snt_end]
        
        temp_wav_lst = wav_lst[snt_id_arr[i]].lower()
        if dataset_name == 'deap':
            parts = temp_wav_lst[:-4].split('_')
            if emotion_state == 'v-a-d' and num_label_dims == 3:
                val = float(parts[-3])
                arousal = float(parts[-2])
                dom = float(parts[-1])
                lab_batch[i, 0] = 1 if val > 5 else 0
                lab_batch[i, 1] = 1 if arousal > 5 else 0
                lab_batch[i, 2] = 1 if dom > 5 else 0
            else:
                if emotion_state == 'arousal':
                    tem = float(parts[-2])
                elif emotion_state == 'dominance':
                    tem = float(parts[-1])
                else:
                    tem = float(parts[-3])
                lab_batch[i] = 1 if tem > 5 else 0
        else:
            lab_batch[i] = int(temp_wav_lst.split('.')[0].split('_')[-1])
    
    inp = torch.from_numpy(sig_batch).float().cuda()
    if lab_batch.ndim == 1:
        lab = torch.from_numpy(lab_batch).long().cuda()
    else:
        lab = torch.from_numpy(lab_batch).float().cuda()
    return inp, lab


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Downstream Training')
    parser.add_argument('--cfg', type=str, default='configs/train_example.cfg', help='Path to config file')
    # parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--pretrained', type=str, default='output/weight/checkpoint_epoch_150.pth', help='Path to pretrained model')
    parser.add_argument('--fold-idx', type=int, default=15, help='Fold index used when list files are not explicit')
    
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
    
    wlen = int(int(options.fs) * int(options.cw_len) / 1000.0)
    wlen_seg_short = wlen // 6
    
    cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))

    arch = options.arch
    multi_label = options.name == 'deap' and getattr(options, 'emotion_state', 'valence') == 'v-a-d'
    label_dims = 3 if multi_label else 1
    
    feature_dim = 128
    
    if arch == 'MultiModalResNet1D':
        option_my = {
            'modalities': {
                'eeg': {'channel_num': channel_num, 'cnn_N_filt': cnn_N_filt,
                       'cnn_len_filt': cnn_len_filt, 'fs': int(options.fs), 
                       'input_dim': wlen_seg_short, 'block': 'BasicBlock', 'structure': 'resnet18'},
            }
        }
        block_types = {'BasicBlock': BasicBlock}
        layers_dict = {'resnet18': [4, 4, 4, 4]}
        segment_encoder = MultiModalResNet1D(option_my, block_types, layers_dict,
                                             num_classes=feature_dim, token_len=32, token_dim=32, attn_heads=4)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Only 'MultiModalResNet1D' is supported.")
    
    checkpoint = torch.load(args.pretrained, map_location='cuda')
    pretrained_dict = checkpoint['CNN_model_par']
    model_dict = segment_encoder.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    segment_encoder.load_state_dict(model_dict)
    
    loaded_keys = set(pretrained_dict.keys())
    missing_keys = set(model_dict.keys()) - loaded_keys
    if missing_keys:
        print(f"Warning: Following layers were not loaded due to size mismatch: {missing_keys}")
    print(f"Successfully loaded {len(loaded_keys)} pretrained layers")
    
    model_num_classes = label_dims if multi_label else int(options.num_classes)
    
    model = MoEEmotionClassifier(
        encoder=segment_encoder,
        segment_length=wlen_seg_short,
        expert_hidden_dims=[64, 128, 256],
        feature_dim=feature_dim,
        num_classes=model_num_classes
    ).cuda()
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    fold_idx = args.fold_idx

    tr_lst = options.tr_lst
    if not tr_lst.endswith('.txt'):
        tr_lst = tr_lst + f'{fold_idx}.txt'
    
    wav_lst_tr = ReadList(tr_lst)
    snt_tr = len(wav_lst_tr)
    
    if multi_label:
        cost = nn.BCEWithLogitsLoss()
    else:
        cost = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(options.lr))
    
    os.makedirs(options.output_folder, exist_ok=True)
    
    for epoch in range(int(options.N_epochs)):
        model.train()
        loss_sum = 0
        
        for i in tqdm(range(int(options.N_batches)), desc=f'Epoch {epoch}'):
            inp, lab = create_batches(int(options.batch_size), options.data_folder + '/',
                                    wav_lst_tr, snt_tr, wlen, channel_num,
                                    channel_num1, channel_num2, channel_lis, options.name,
                                    getattr(options, 'emotion_state', 'valence'),
                                    num_label_dims=label_dims)
            
            trial_logits, _, _ = model(inp)
            if multi_label:
                lab_float = lab.float()
                loss = cost(trial_logits, lab_float)
            else:
                loss = cost(trial_logits, lab)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
        
        print(f'Epoch {epoch}, Loss: {loss_sum / int(options.N_batches):.4f}')
        
        if epoch % 10 == 0:
            checkpoint = {'CNN_model_par': model.state_dict()}
            torch.save(checkpoint, os.path.join(options.output_folder, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()

