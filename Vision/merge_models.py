import copy
import numpy as np
import torch
import torch.nn as nn

dataset_list = ['MNIST', 'EuroSAT', 'GTSRB', 'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(model, names, params):
    for name, p in zip(names, params):
        set_attr(model, name.split("."), p)
        # # if you want to get state_dict or named_parameters, you can use the following code,
        # # but it seems that the alphas are not trained for this version
        # set_attr(model, name.split("."), nn.Parameter(p))


def load_weights_with_param(model, names, params):
    for name, p in zip(names, params):
        set_attr(model, name.split("."), nn.Parameter(p))


class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, granularity, init_scale=0.3):
        super(MergedModel, self).__init__()

        self.models = models
        self.granularity = granularity
        self.pretrained_model = pretrained_model
        self.init_scale = init_scale

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()

        for model in self.models:
            alpha = nn.ParameterList()
            if self.granularity == 'taskwise':
                alpha.append(nn.Parameter(torch.tensor(
                    self.init_scale), requires_grad=True))
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.tensor(
                        self.init_scale), requires_grad=True))
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.ones_like(
                        param) * self.init_scale, requires_grad=True))
            else:
                raise NotImplementedError(
                    f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        self.merged_model = copy.deepcopy(self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())
                     [name] - pretrained_param)
            param += pretrained_param
            merged_param.append(param)

        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def forward(self, x):
        merged_model = self.get_merged_model()
        return merged_model(x)

    def get_avg_coef(self):
        res = {}
        for idx, alpha in enumerate(self.alphas):
            mean_coef = np.mean([a.mean().item() for a in alpha])
            res[dataset_list[idx]] = mean_coef
        return res

    def get_max_coef(self):
        res = {}
        for idx, alpha in enumerate(self.alphas):
            max_coef = np.max([a.max().item() for a in alpha])
            res[dataset_list[idx]] = max_coef
        return res

    def get_min_coef(self):
        res = {}
        for idx, alpha in enumerate(self.alphas):
            min_coef = np.min([a.min().item() for a in alpha])
            res[dataset_list[idx]] = min_coef
        return res

    def get_avg_coef_per_param(self):
        assert self.granularity in ['layerwise', 'elementwise']
        res = {}
        for k in range(len(self.models)):
            alpha_dict = {}
            for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
                    alpha = self.alphas[k][idx]
                    alpha_dict[name] = alpha.mean().mean().item()
            res[dataset_list[k]] = alpha_dict
        
        return res


