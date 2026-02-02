import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import MultiModalResNet1D
except ImportError:
    from ..models import MultiModalResNet1D


class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, encoder_q=None, encoder_k=None):
        super(ModelMoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].t()
            self.queue[:, :batch_size - remaining] = keys[remaining:].t()
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.t()
            ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        _, q, *rest = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            _, k, *rest = self.encoder_k(im_k_)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        with torch.no_grad():
            self._momentum_update_key_encoder()

        loss, q, k = self.contrastive_loss(im1, im2)
        self._dequeue_and_enqueue(k)

        return loss

