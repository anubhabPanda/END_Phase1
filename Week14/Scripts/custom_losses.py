import torch
from torch import nn
import torch.nn.functional as F


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)

        count = (targets != self.pad_index).sum().item()

        return batch_loss, count

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean', ignore_index = -100):
        super().__init__()
        self.ε,self.reduction = ε,reduction
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 