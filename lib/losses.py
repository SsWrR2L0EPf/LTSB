import torch
import torch.nn as nn
import torch.nn.functional as F


class LTSBLoss(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list)
        weight = torch.log(cls_num_list / cls_num_list.sum() + 1e-9)[None, :]
        self.weight = weight.cuda()
        w = (cls_num_list.mean() / cls_num_list) ** 0.25
        self.w = w.cuda()

    def cls_loss(self, q, k, p, logits, ls, labels):
        logits = logits + self.weight - torch.max(logits, 1, True)[0].detach()
        cross_entropy_loss = F.cross_entropy(logits, labels)

        sim = (1 - (F.cosine_similarity(q, k)).mean()) * 2 \
              + (1 - (F.cosine_similarity(p, k) * ls).mean())

        loss = cross_entropy_loss + sim
        return loss

    def enc_loss(self, q, k, p, logits1, logits2, labels):
        logits1 = logits1 + self.weight - torch.max(logits1, 1, True)[0].detach()
        logits2 = logits2 + self.weight - torch.max(logits2, 1, True)[0].detach()
        cross_entropy_loss = F.cross_entropy(logits1, labels) \
                             + F.cross_entropy(logits2, labels)

        w = self.w[labels]
        sim = (1 - (F.cosine_similarity(q, k)).mean()) * 2 \
              + (1 - (F.cosine_similarity(p, k) * w / w.sum()).sum()) * 2

        loss = cross_entropy_loss + sim

        return loss

    def forward(self, q, k=None, p=None, logits=None, ls=None, labels=None):
        if ls.ndim == 1:
            loss = self.cls_loss(q, k, p, logits, ls, labels)
        else:
            loss = self.enc_loss(q, k, p, logits, ls, labels)

        return loss
