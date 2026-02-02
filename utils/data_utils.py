import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset
import os


def ReadList(list_file):
    with open(list_file, "r") as f:
        lines = f.readlines()
    return [x.rstrip() for x in lines]


def read_conf(path_cfg):
    cfg_file = path_cfg
    class Options:
        pass
    options = Options()
    
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    options.name = Config.get('data', 'name')
    options.tr_lst = Config.get('data', 'tr_lst')
    options.te_lst = Config.get('data', 'te_lst')
    options.data_folder = Config.get('data', 'data_folder')
    options.output_folder = Config.get('data', 'output_folder')
    options.pt_file = Config.get('data', 'pt_file')
    options.segment_or_not = Config.get('data', 'segment_or_not')
    try:
        options.emotion_state = Config.get('data', 'emotion_state')
    except:
        options.emotion_state = 'valence'

    options.fs = Config.get('windowing', 'fs')
    options.cw_len = Config.get('windowing', 'cw_len')
    options.cw_shift = Config.get('windowing', 'cw_shift')
    options.channel_num = Config.get('windowing', 'channel_num')

    options.arch = Config.get('cnn', 'arch')
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')
    options.mulhead_num_hiddens = Config.get('cnn', 'mulhead_num_hiddens')
    options.mulhead_num_heads = Config.get('cnn', 'mulhead_num_heads')
    options.mulhead_num_query = Config.get('cnn', 'mulhead_num_query')
    options.dropout_fc = Config.get('cnn', 'dropout_fc')
    options.att_hidden_dims_fc = Config.get('cnn', 'att_hidden_dims_fc')
    options.hidden_dims_fc = Config.get('cnn', 'hidden_dims_fc')
    options.num_classes = Config.get('cnn', 'num_classes')

    options.lr = Config.get('optimization', 'lr')
    options.batch_size = Config.get('optimization', 'batch_size')
    options.N_epochs = Config.get('optimization', 'N_epochs')
    options.N_batches = Config.get('optimization', 'N_batches')
    options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
    options.seed = Config.get('optimization', 'seed')
    options.fold = Config.get('optimization', 'fold')
    options.patience = Config.get('optimization', 'patience')

    return options


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


class EEGDataset(Dataset):
    def __init__(self, dataset_list_file, data_folder, dataset_name, wlen, channel_num,
                 channel_num1=0, channel_num2=None, channel_lis=None, emotion_state='valence'):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.wlen = wlen
        self.channel_num = channel_num
        self.channel_num1 = channel_num1
        self.channel_num2 = channel_num2
        self.channel_lis = channel_lis
        self.emotion_state = emotion_state
        
        with open(dataset_list_file, 'r') as f:
            self.audio_list = [line.strip() for line in f.readlines()]

    def _create_batches_rnd(self, filename):
        data = scipy.io.loadmat(filename)
        
        if self.dataset_name == 'seed':
            signal = data['eeg']
            fs = 100
        else:
            signal = data['data']
            fs = 128
        
        signal = signal[:, :]
        
        if self.channel_lis is None or len(self.channel_lis) <= 2:
            signal = signal[self.channel_num1:self.channel_num2, :]
        else:
            signal = signal[self.channel_lis, :]

        snt_len = signal.shape[-1]
        min_len = self.wlen * 3
        
        if snt_len < min_len:
            pad_width = min_len - snt_len
            if signal.ndim == 1:
                signal = np.pad(signal, (0, pad_width), mode='constant')
            else:
                signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
            snt_len = min_len

        snt_beg = np.random.randint(snt_len - self.wlen * 2 - 1)
        snt_end = snt_beg + self.wlen

        sig_batch1 = signal[:, snt_beg:snt_end].squeeze()
        sig_batch2 = signal[:, snt_beg + self.wlen:snt_end + self.wlen].squeeze()

        inp1 = torch.from_numpy(sig_batch1).float()
        inp2 = torch.from_numpy(sig_batch2).float()
        
        temp_wav_lst = filename.split('/')[-1].lower()
        
        if self.dataset_name == 'deap':
            subject_id = int(temp_wav_lst.split('_')[0])
            parts = temp_wav_lst[:-4].split('_')
            if self.emotion_state == 'arousal':
                tem = float(parts[-2])
            elif self.emotion_state == 'dominance':
                tem = float(parts[-1])
            else:
                tem = float(parts[-3])
            lab = 1 if tem > 5 else 0
        elif self.dataset_name == 'dreamer':
            subject_id = int(temp_wav_lst.split('_')[0])
            parts = temp_wav_lst[:-4].split('_')
            if self.emotion_state == 'arousal':
                tem = int(parts[-2])
            elif self.emotion_state == 'dominance':
                tem = int(parts[-1])
            else:
                tem = int(parts[-3])
            lab = 1 if tem >= 3 else 0
        else:
            subject_id = int(temp_wav_lst.split('_')[0])
            lab = int(temp_wav_lst.split('.')[0].split('_')[-1])
        
        return inp1, inp2, lab, subject_id - 1

    def __getitem__(self, index):
        audio_name = self.audio_list[index]
        dataFile = os.path.join(self.data_folder, audio_name)
        sinc_input1, sinc_input2, lab, subject_num = self._create_batches_rnd(dataFile)
        return sinc_input1, sinc_input2, lab, subject_num

    def __len__(self):
        return len(self.audio_list)

