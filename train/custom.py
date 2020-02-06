import torch.nn.functional as F
import torch.nn as nn
import torch

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None, token_dim=-1,
            sequence_dim=-2):

        #normalize by token classes and guess most probable sequence
        prediction = F.softmax(prediction, token_dim)\
                .argmax(sequence_dim)


        scores = (prediction == target)
        n_padded = 0
        if mask is not None:
            n_padded = (mask == 0).sum()
        return scores.sum() / float(scores.numel() - n_padded)


def smooth_cross_entropy(prediction, target, eps=0.1,
        ignore_index=0):


    mask = (target == ignore_index).unsqueeze(-1)

    prediction = prediction.transpose(1,2)
    n_classes = prediction.shape[-1]
    #one hot encode target
    p = F.one_hot(target, n_classes)
    #uniform distribution probability
    u = 1.0 / n_classes
    p_prime = (1.0 - eps) * p + eps * u
    #ignore padding indices
    p_prime = p_prime.masked_fill(mask, 0)
    #cross entropy
    h = -torch.sum(p_prime * F.log_softmax(prediction, -1))
    #mean reduction
    n_items = torch.sum(target != ignore_index)

    return h / n_items


class TFSchedule:
    """
    From https://www.tensorflow.org/tutorials/text/transformer. Wrapper for Optimizer, gradually increases learning rate for a warmup period before learning rate decay sets in.
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):

        self.opt = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        self._step = 0
        self._rate = 0

    def step(self):

        self._step += 1
        rate = self.rate()
        for p in self.opt.param_groups:
            p['lr'] = rate

        self._rate = rate
        self.opt.step()


    def rate(self, step=None):

        if step is None:
            step = self._step

        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.d_model ** (-0.5) * min(arg1, arg2)

