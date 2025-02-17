import copy
import random
import re
import os
import torch
import torch.nn as nn
from utils.load_config import cache_dir
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


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


def del_ex(model, exclude):
    new_model = copy.deepcopy(model)
    for param_name, param_value in model.named_parameters():
        exc = [re.match(regex, param_name) for regex in exclude]
        if any(exc):
            del_attr(new_model, param_name.split("."))
    return new_model


class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, granularity):
        super(MergedModel, self).__init__()
        self.pretrained_model = pretrained_model
        # self.models = copy.deepcopy(models)
        self.models = models
        self.granularity = granularity

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            if self.granularity == 'taskwise':
                alpha.append(nn.Parameter(
                    torch.tensor(0.3), requires_grad=True))
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(
                        torch.tensor(0.3), requires_grad=True))
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.ones_like(
                        param) * 0.3, requires_grad=True))
            else:
                raise NotImplementedError(
                    f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        self.merged_model = copy.deepcopy(
            self.pretrained_model)
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
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param.append(param)

        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def get_named_parameters(self):
        merged_param = {}
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param[name] = param
        return merged_param

    def forward(self, x):
        merged_model = self.get_merged_model()
        # if isinstance(x, dict):
        if hasattr(x, 'keys'):
            return merged_model(**x)
        else:
            return merged_model(x)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class LabeledDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_indices = []
        for i, dataset in enumerate(datasets):
            self.dataset_indices.extend(
                [(i, idx) for idx in range(len(dataset))])
        random.shuffle(self.dataset_indices)

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.dataset_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        return sample, dataset_idx


def custom_collate_fn(batch):
    # Custom collate function to handle varying input sizes
    data = [item[0] for item in batch]
    source_loader = torch.tensor([item[1] for item in batch])
    return {'data': data, 'source_loader': source_loader}


def merge_data_loaders_from_trainers(trainers, batch_size=16, num_workers=0):
    # Extract datasets from the data loaders
    datasets = []
    for trainer in trainers:
        dataloader = trainer.get_train_dataloader()
        dataset = []
        for item in dataloader:
            dataset.append(trainer._prepare_inputs(item))
        datasets.append(dataset)

    # Create a merged dataset
    merged_dataset = LabeledDataset(datasets)

    # Create a new data loader from the merged dataset
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return merged_loader


class TransformedDataDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def transformed_data_collate_fn(batch):
    # data = torch.stack([item[0] for item in batch])
    # source_loaders = torch.tensor([item[1] for item in batch])
    # attention_mask = torch.stack([item[2] for item in batch])
    data = batch[0][0]
    source_loaders = batch[0][1]
    attention_mask = batch[0][2]
    return {'data': data, 'source_loader': source_loaders, 'attention_mask': attention_mask}


def transform_data_loader_prelayer(data_loader, model, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            # output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            output = model(x)

            # batchsize = 1
            transformed_data.append(
                (output[0].cpu(), source_loader, output[1].cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_prelayer_pertask(data_loader, merged_model, models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            inputs = []
            attention_masks = []

            # model_output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # model_output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            model_output = merged_model(x)
            inputs.append(model_output[0])
            attention_masks.append(model_output[1])

            model = models[source_loader.item()]
            model_output = model(x)
            inputs.append(model_output[0])
            attention_masks.append(model_output[1])

            # shape of inputs: [2, batch_size, seq_length, embedding_dim] -> [batch_size, 2, seq_length, embedding_dim]
            # shape of attention_masks: [2, batch_size, 1, seq_length, seq_length] -> [batch_size, 2, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            if attention_masks[0] is not None:
                attention_masks = torch.stack(
                    attention_masks).permute(1, 0, 2, 3, 4).cpu()
            else:
                attention_masks = torch.zeros(
                    inputs.shape[0], inputs.shape[1], 1, inputs.shape[2], inputs.shape[2])

            # batchsize = 1
            transformed_data.append((inputs, source_loader, attention_masks))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_layer_pertask(data_loader, merged_model, models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # shape of x: [batch_size, 2, seq_length, embedding_dim] -> [2, batch_size, seq_length, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            # shape of attention mask: [batch_size, 2, 1, seq_length, seq_length] -> [2, batch_size, 1, seq_length, seq_length]
            attention_mask = data['attention_mask'].to(device)
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4)

            inputs = []

            output = merged_model(
                x[0], attention_mask[0], None, None, None, None, False)[0]
            inputs.append(output)

            model = models[source_loader.item()]
            output = model(x[1], attention_mask[1], None,
                           None, None, None, False)[0]
            inputs.append(output)

            # shape of inputs: [2, batch_size, seq_length, embedding_dim] -> [batch_size, 2, seq_length, embedding_dim]
            # shape of attention_masks: [2, batch_size, 1, seq_length, seq_length] -> [batch_size, 2, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4).cpu()

            # batchsize = 1
            transformed_data.append(
                (inputs, source_loader, attention_mask.cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def create_pre_encoder_activation_extractor(model, language_model_name):
    def pre_encoder_activation_extractor(x):
        class StopForwardPass(Exception):
            pass

        encoder_input = []

        def create_hook():
            def hook(module, input, output):
                # wrong here
                encoder_input.append(input)
                raise StopForwardPass
            return hook

        if language_model_name == 'bert-base-uncased':
            block = model.bert.encoder.layer[0]
        elif language_model_name == 'roberta-base':
            block = model.roberta.encoder.layer[0]
        handle = block.register_forward_hook(create_hook())

        try:
            model(**x)
        except StopForwardPass:
            pass

        handle.remove()

        return encoder_input[0]

    return pre_encoder_activation_extractor


def load_pretrained_model(args):
    try:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)

    return pretrained_model


def load_fine_tuned_model(args, dataset_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.load_model_paths_dict[dataset_name]).to(args.device)
    return model

# This function also merged the classification head, which should be reloaded for each dataset


def load_avg_merged_model(args, merge_coef=0.3):
    pretrained_model = load_pretrained_model(args)

    new_state_dict = pretrained_model.state_dict()

    for dataset in args.dataset_names:
        model = load_fine_tuned_model(args, dataset)
        for name, param in pretrained_model.named_parameters():
            if 'classifier' not in name:
                new_param = (dict(model.named_parameters())[
                             name]-dict(pretrained_model.named_parameters())[name]) * merge_coef
                new_state_dict[name] = new_state_dict[name] + new_param
        del model
        torch.cuda.empty_cache()

    pretrained_model.load_state_dict(new_state_dict)
    return pretrained_model


def load_merged_layers(args, layer_idx, granularity='elementwise'):
    pretrained_model = load_pretrained_model(args)

    if args.language_model_name == 'bert-base-uncased':
        layer_pretrained = pretrained_model.bert.encoder.layer[layer_idx]
    elif args.language_model_name == 'roberta-base':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]

    layers = []
    for dataset in args.dataset_names:
        model = load_fine_tuned_model(args, dataset)
        if args.language_model_name == 'bert-base-uncased':
            layer = model.bert.encoder.layer[layer_idx]
        elif args.language_model_name == 'roberta-base':
            layer = model.roberta.encoder.layer[layer_idx]
        layers.append(layer)
        del model
        torch.cuda.empty_cache()

    merged_layers = MergedModel(layer_pretrained, layers, granularity)

    del pretrained_model
    torch.cuda.empty_cache()

    return merged_layers, layers


def load_pretrained_layer(args, layer_idx):
    pretrained_model = load_pretrained_model(args)

    if args.language_model_name == 'bert-base-uncased':
        layer_pretrained = pretrained_model.bert.encoder.layer[layer_idx]
    elif args.language_model_name == 'roberta-base':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]

    return layer_pretrained



def new_forward_bert(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                     head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                     output_hidden_states=None, return_dict=None):
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]
    # pooled_output.shape = [batchsize, 768]
    return pooled_output


def new_forward_roberta(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                        output_hidden_states=None, return_dict=None):
    outputs = self.roberta(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]
    # sequence_output.shape = [batchsize, seqlength, 768]
    return sequence_output