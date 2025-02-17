
import torch
import torch.nn as nn
from collections import defaultdict
from src.datasets.common import maybe_dictionarize
from tqdm import tqdm
import re


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])



def fisher_merging(models_to_merge, val_loaders, exclude_param_names_regex, fisher_scaling_coefficient, device, 
                   normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
    def get_param_squared_gradients(model: nn.Module, param_names_to_merge: list):
        """
        get the squared gradients of parameters
        :param model: nn.Module, model
        :param param_names_to_merge: list, list of parameter names that need to be merged
        :return:
        """
        param_squared_gradients = {param_name: param_value.grad.detach(
        ) ** 2 for param_name, param_value in model.named_parameters() if param_name in param_names_to_merge}
        return param_squared_gradients

    def get_models_fisher_norm(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list):
        """
        get normalization of fisher weights of all the models that need to be merged
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :return:
        """
        # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
        models_fisher_norm_dict = {}
        # compute L2 norm over models for each parameter
        for param_name, _ in models_to_merge_param_dict.items():
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
            models_fisher = torch.stack([model_to_merge_fisher_weights[param_name]
                                        for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0)
            dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
            # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter

            if len(dims) >2:
                print(param_name)
                models_fisher = models_fisher.view(models_fisher.shape[0], -1)
                dims = [1]

            models_fisher_norm = torch.norm(models_fisher, dim=dims)

            models_fisher_norm_dict[param_name] = models_fisher_norm

        # Tensor, shape (num_models_to_merge, num_parameters)
        models_fisher_norm = torch.stack(
            [models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()], dim=1)
        # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
        models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
        return models_fisher_norm

    def merging_with_fisher_weights(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list, fisher_scaling_coefficients: torch.Tensor,
                                    normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
        """
        merge parameters of different models with computed fisher weights
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :param fisher_scaling_coefficients: torch.Tensor, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :return:
        """
        # dict, dictionary of model parameters
        merged_params = {}

        if normalize_fisher_weight:
            # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
            models_fisher_norm = get_models_fisher_norm(models_to_merge_param_dict=models_to_merge_param_dict,
                                                        models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list)

        for param_name, param_value_list in models_to_merge_param_dict.items():
            # shape (num_models_to_merge, *parameter_shape)
            param_values = torch.stack(param_value_list, dim=0)
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
            models_to_merge_fisher_weights = torch.stack([model_to_merge_fisher_weights[param_name]
                                                          for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0) + minimal_fisher_weight

            # Tensor, shape (num_models_to_merge, 1, 1, ...)
            reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(
                -1, *[1 for _ in range(param_values.dim() - 1)]).to(param_values.device)

            if normalize_fisher_weight:
                # Tensor, shape (num_models_to_merge, )
                _models_fisher_norm = 1.0 / \
                    (models_fisher_norm + minimal_fisher_weight)
                normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(
                    -1, *[1 for _ in range(param_values.dim() - 1)])
                reshaped_scaling_coefficients = reshaped_scaling_coefficients * \
                    normalized_models_fisher_norm

            # shape (*parameter_shape)
            numerator = (reshaped_scaling_coefficients *
                         models_to_merge_fisher_weights * param_values).sum(dim=0)

            # shape (*parameter_shape)
            denominator = (reshaped_scaling_coefficients *
                           models_to_merge_fisher_weights).sum(dim=0)

            merged_param = numerator / denominator
            merged_params[param_name] = merged_param
        return merged_params

    # dictionary of list, where key is the parameter name,
    # value is a list of the corresponding parameters of all the models that need to be merged
    models_to_merge_param_dict = defaultdict(list)

    # list of dictionaries with length len(models_to_merge),
    # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
    models_to_merge_fisher_weights_list = []


    for model_idx, (model_to_merge, val_loader) in enumerate(zip(models_to_merge, val_loaders)):
        param_dict = {param_name: param_value for param_name,
                      param_value in model_to_merge.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(
            param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

        for param_name in param_names_to_merge:
            models_to_merge_param_dict[param_name].append(
                param_dict[param_name])

        # list of dictionaries with length (num_fisher_examples // batch_size) or (num_fisher_examples // batch_size) + 1,
        # each dictionary records the fisher weights of parameters for model_to_merge computed by examples in a batch
        batches_fisher_weights_list = []

        num_computed_examples = 0

        for _, data in tqdm(enumerate(val_loader), desc=f"computing fisher weights for model {model_idx}"):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            logits = model_to_merge(x)

            # use detach() to detach from the computation graph
            # Tensor, shape (batch_size, num_label_classes)
            labels_probabilities = torch.softmax(
                logits, dim=-1).detach()
            labels_log_probabilities = torch.log_softmax(
                logits, dim=-1)
            # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
            labels_expectations = torch.sqrt(
                labels_probabilities) * labels_log_probabilities
            # sum over label classes and batch dimension
            sum_labels_expectations = labels_expectations.sum(
                dim=-1).sum(dim=0)
            model_to_merge.zero_grad()
            sum_labels_expectations.backward()
            # dict, fisher weights of a batch
            batch_fisher_weights = get_param_squared_gradients(
                model=model_to_merge, param_names_to_merge=param_names_to_merge)

            batches_fisher_weights_list.append(batch_fisher_weights)
            num_computed_examples += x.size(0)

        model_to_merge_fisher_weights = {}
        for batch_fisher_weights in batches_fisher_weights_list:
            for key in batch_fisher_weights:
                if key not in model_to_merge_fisher_weights:
                    model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                else:
                    model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        # mean over batches
        for key in model_to_merge_fisher_weights:
            model_to_merge_fisher_weights[key] /= num_computed_examples
        models_to_merge_fisher_weights_list.append(
            model_to_merge_fisher_weights)
        
    fisher_scaling_coefficients = [fisher_scaling_coefficient] * len(models_to_merge)

    # merging with fisher weights
    # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
    if fisher_scaling_coefficients is None:
        fisher_scaling_coefficients = torch.ones(
            len(models_to_merge)) / len(models_to_merge)
    else:
        assert isinstance(fisher_scaling_coefficients,
                          list), "wrong type of fisher_scaling_coefficients, should be list!"
        assert len(fisher_scaling_coefficients) == len(
            models_to_merge), "mismatched length of fisher_scaling_coefficients!"
        fisher_scaling_coefficients = torch.Tensor(
            fisher_scaling_coefficients)
    # merging with fisher weights
    merged_params = merging_with_fisher_weights(models_to_merge_param_dict=models_to_merge_param_dict, models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
                                                fisher_scaling_coefficients=fisher_scaling_coefficients, normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight)

    return merged_params
