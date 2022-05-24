"""https://github.com/facebookresearch/moco"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import deque
import random
import math


def flatten(t):
    return t.reshape(t.shape[0], -1)


class LTSB_CLS(nn.Module):

    def __init__(self, base_encoder, dim=128, m=0.999, num_classes=1000, buffer_size=16):
        super().__init__()

        self.m = m
        self.num_classes = num_classes
        self.dim = dim

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.buffer_size = buffer_size
        self.register_buffer('conf', torch.zeros(num_classes, buffer_size, dtype=torch.float32))
        self.register_buffer('d', F.normalize(torch.rand((num_classes, buffer_size, dim), dtype=torch.float32), dim=2))
        self.register_buffer('ptr', torch.zeros(num_classes, dtype=torch.long))

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.linear = nn.Linear(dim_mlp, num_classes)
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp),
                                          nn.ReLU(True),
                                          self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp),
                                          nn.ReLU(True),
                                          self.encoder_k.fc)

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

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _train(self, im_q, im_k, target, epoch):
        q = self.encoder_q(im_q)
        q = F.normalize(q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k)

        logits_q = self.linear(self.feat_after_avg_q)
        with torch.no_grad():
            if epoch < 0:
                pred = target
            else:
                pred = torch.argmax(logits_q, dim=1)
            d = []
            conf = []
            prob = torch.sigmoid(logits_q).gather(1, pred.unsqueeze(1)).squeeze(1)
            for i in range(pred.size(0)):
                label = pred[i].item()
                index = random.randint(0, self.buffer_size - 1)
                p = self.d[label][index]
                d.append(p)
                conf.append(self.conf[label][index])

            pred_gather = concat_all_gather(pred)
            prob_gather = concat_all_gather(prob)
            k_gather = concat_all_gather(k)
            for i in range(pred_gather.size(0)):
                label = pred_gather[i].item()
                ptr = self.ptr[label]
                self.d[label][ptr].copy_(k_gather[i].cpu())
                self.conf[label][ptr] = prob_gather[i].item()
                self.ptr[label] = (ptr + 1) % self.buffer_size
            d = torch.stack(d).cuda()
            conf = torch.tensor(conf).cuda()

        return q, k, d, logits_q, conf

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
