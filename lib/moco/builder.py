"""https://github.com/facebookresearch/moco"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision
from collections import deque
import random
import math


def flatten(t):
    return t.reshape(t.shape[0], -1)


class LTSB_CLS(nn.Module):

    def __init__(self, base_encoder, dim=128, m=0.999, num_classes=1000, buffer_size=16, K=2048):
        super().__init__()

        self.m = m
        self.num_classes = num_classes
        self.dim = dim
        self.K = K

		self.encoder_q = base_encoder(num_classes=dim)
		self.encoder_k = base_encoder(num_classes=dim)

        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.linear = nn.Linear(dim_mlp, num_classes)
        inner_dim = dim_mlp * 2
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, inner_dim, bias=False), nn.BatchNorm1d(inner_dim),
                                          nn.ReLU(True),
                                          nn.Linear(inner_dim, dim))
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, inner_dim, bias=False), nn.BatchNorm1d(inner_dim),
                                          nn.ReLU(True),
                                          nn.Linear(inner_dim, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # cross_entropy
        self.layer = -2
        # self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self._register_hook()

    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_q(self, _, input, output):
        # self.feat_before_avg_q = input[0]
        self.feat_after_avg_q = flatten(output)
        self.feat_before_avg_q = output.detach()

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, im_q, im_k, target, epoch):
        bs = im_q.size(0)
        im = torch.cat([im_q, im_k], dim=0)
        target = torch.cat([target, target], dim=0)
        #q2 = self.encoder_q(im_k)
        #q2 = F.normalize(q2)
        #f2 = self.feat_after_avg_q
        q = self.encoder_q(im)
        q = F.normalize(q)
        logits = self.linear(self.feat_after_avg_q)
        logits, logits2 = logits.chunk(2)

        with torch.no_grad():
            self._momentum_update_key_encoder(epoch)
            #im_k, target, idx_unshuffle = self._batch_shuffle_ddp(im_k, target)
            k = self.encoder_k(im)
            k = F.normalize(k)
            #k, labels = self._batch_unshuffle_ddp(k, target, idx_unshuffle)

            #im_k2, target2, idx_unshuffle2 = self._batch_shuffle_ddp(im_k, target)
            #k2 = self.encoder_k(im_q)
            #k2 = F.normalize(k2)
            #k2, labels2 = self._batch_unshuffle_ddp(k2, target2, idx_unshuffle2)

            feat = self.feat_before_avg_q
            weight = self.linear.weight[target].view(target.size(0), self.linear.in_features, 1, 1)
            cam = (feat * weight).sum(1, True)
            mx, mn = cam.max(0, True)[0], cam.min(0, True)[0]
            cam = (cam - mn) / (mx - mn)
            cam = F.interpolate(cam, im.shape[2:], mode='bilinear', align_corners=False)
            #im_mask = im_q * (cam < 0.7)
            im_mask = im * (1 - cam)

            m = self.encoder_k(im_mask)
            m = F.normalize(m)

        #qs = torch.cat([q, q2], dim=0)
        #ks = torch.cat([k2, k], dim=0)
        que_k = torch.cat([k, self.queue], dim=0)
        #que_target = torch.cat([target, self.queue_l], dim=0)
        l_pos = (q @ que_k.T)
        l_posm = (q * m).sum(1, True)
        #l_que = (q @ self.queue.detach().t())
        with torch.no_grad():
            targets_pos = (target.unsqueeze(1) == target.unsqueeze(0))
            targets_pos = targets_pos / targets_pos.sum(1, True) /3
            targets_posm = torch.ones_like(l_posm, dtype=torch.float32) /3
            targets_que = (target.unsqueeze(1) == self.queue_l.unsqueeze(0))
            targets_que = targets_que / (targets_que.sum(1, True) + 1e-5) /3
            targets_con = torch.cat([targets_pos, targets_que, targets_posm], dim=1)
        sim = torch.cat([l_pos, l_posm], dim=1) / 0.2
        '''l_pos = (q * k).sum(1, True)
        l_posm = (q * m).sum(1, True)
        l_neg = (q @ self.queue.clone().detach().t())
        sim = torch.cat([l_pos, l_posm, l_neg], dim=1)/0.07
        with torch.no_grad():
            targets_pos = torch.ones_like(l_pos, dtype=torch.float32) / 3
            targets_posm = torch.ones_like(l_posm, dtype=torch.float32) / 3
            targets_neg = ((target.unsqueeze(1) == self.queue_l.unsqueeze(0)) &
                           (l_neg < 0.8)).float()
            targets_neg = targets_neg / (targets_neg.sum(1, True) + 1) / 3
            targets_con = torch.cat([targets_pos, targets_posm, targets_neg], dim=1)'''

        #self._dequeue_and_enqueue(k, target)
        if epoch < 60:
            self._dequeue_and_enqueue(k, target)
        else:
            self._dequeue_and_enqueue(k, torch.argmax(logits, dim=1))

        return logits, logits2, sim, targets_con

    def _inference(self, image):
        self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)

        return encoder_q_logits

    def forward(self, im_q, im_k=None, target=None, epoch=None):
        if self.training and im_k is not None:
            return self._train(im_q, im_k, target, epoch)
        else:
            return self._inference(im_q)


class LTSB_ENC(nn.Module):

    def __init__(self, base_encoder, dim=128, m=0.999, num_classes=1000):
        super().__init__()

        self.m = m
        self.num_classes = num_classes
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        if base_encoder == torchvision.models.resnet152:
            self.encoder_q = base_encoder(num_classes=dim, pretrained=True)
            self.encoder_q2 = base_encoder(num_classes=dim, pretrained=True)
            self.encoder_k = base_encoder(num_classes=dim, pretrained=True)
        else:
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_q2 = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.linear = nn.Linear(dim_mlp, num_classes)
        self.linear2 = nn.Linear(dim_mlp, num_classes)
        self.encoder_q.fc = nn.Identity()
        self.encoder_q2.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()
        self.proj_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp),
                                    nn.ReLU(True),
                                    nn.Linear(dim_mlp, dim))
        self.proj_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp),
                                    nn.ReLU(True),
                                    nn.Linear(dim_mlp, dim))

        for param_q, param_q2 in zip(self.encoder_q.parameters(), self.encoder_q2.parameters()):
            param_q2.data.copy_(param_q.data)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # cross_entropy
        self.layer = -2
        # self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self.feat_after_avg_q2 = None
        self._register_hook()

    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_q(self, _, input, output):
        # self.feat_before_avg_q = input[0]
        self.feat_after_avg_q = flatten(output)

    def _hook_q2(self, _, input, output):
        # self.feat_before_avg_q2 = input[0]
        self.feat_after_avg_q2 = flatten(output)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        layer_q.register_forward_hook(self._hook_q)

        layer_q2 = self._find_layer(self.encoder_q2)
        assert layer_q2 is not None, f'hidden layer ({self.layer}) not found'
        layer_q2.register_forward_hook(self._hook_q2)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_q2, param_k in zip(self.encoder_q.parameters(), self.encoder_q2.parameters(),
                                              self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + (param_q.data + param_q2.data) / 2 * (1. - self.m)
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _train(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = self.proj_q(q)
        q = F.normalize(q)

        q2 = self.encoder_q2(im_q)
        q2 = self.proj_q(q2)
        q2 = F.normalize(q2)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.proj_k(k)
            k = F.normalize(k)

        logits_q = self.linear(self.feat_after_avg_q)
        logits_q2 = self.linear2(self.feat_after_avg_q2)

        return q, k, q2, logits_q, logits_q2

    def _inference(self, image):
        self.encoder_q(image)
        self.encoder_q2(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q) + self.linear2(self.feat_after_avg_q2)

        return encoder_q_logits

    def forward(self, im_q, im_k=None, target=None, epoch=None):
        if self.training and im_k is not None:
            return self._train(im_q, im_k)
        else:
            return self._inference(im_q)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
