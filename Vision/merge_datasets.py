import random
import torch
from torch.utils.data import Dataset, DataLoader


class MergedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_indices = []
        for i, dataset in enumerate(datasets):
            self.dataset_indices.extend([(i, idx)
                                        for idx in range(len(dataset))])
        random.shuffle(self.dataset_indices)

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.dataset_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        return sample, dataset_idx


def custom_collate_fn(batch):
    data = torch.stack([item[0][0] for item in batch])
    targets = torch.tensor([item[0][1] for item in batch])
    source_loaders = torch.tensor([item[1] for item in batch])
    return {'data': data, 'target': targets, 'source_loader': source_loaders}


def merge_data_loaders(data_loaders, batch_size=128, num_workers=1, shuffle=True):
    # Extract datasets from the data loaders
    datasets = [data_loader.dataset for data_loader in data_loaders]
    datasize = [len(dataset) for dataset in datasets]

    # Create a merged dataset
    merged_dataset = MergedDataset(datasets)

    # Create a new data loader from the merged dataset
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    data = torch.stack([item[0] for item in batch])
    source_loaders = torch.tensor([item[1] for item in batch])
    return {'data': data, 'source_loader': source_loaders}


def transform_data_loader(data_loader, model, device, transpose=False, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'].to(device)
            source_loader = data['source_loader']

            if not transpose:
                output = model(x)
            else:
                output = model(x.permute(1, 0, 2))

            # the original output has shape (seq_len, batch_size, num_features)
            output = output.permute(1, 0, 2)

            output = output.cpu()

            for i in range(output.size(0)):
                transformed_data.append((output[i], source_loader[i]))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=data_loader.batch_size,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_per_task_pre_resblock(data_loader, merged_model, task_models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'].to(device)
            source_loader = data['source_loader']

            output = []
            output.append(merged_model(x))
            for task_model in task_models:
                output.append(task_model(x))

            # output[i]: (seq_len, batch_size, num_features) -> (batch_size, seq_len, num_features)
            for i in range(len(output)):
                output[i] = output[i].permute(1, 0, 2).cpu()

            # each datapoint consists of (torch.Tensor([output_merged, output_task_io]), source_loader)
            for i in range(output[0].size(0)):
                data_list = []
                data_list.append(output[0][i])
                data_list.append(output[source_loader[i]+1][i])
                data_tensor = torch.stack(data_list)
                transformed_data.append((data_tensor, source_loader[i]))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=data_loader.batch_size,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_per_task(data_loader, merged_model, task_models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # shape x: (batchsize, 2, seq_len, num_features)
            x = data['data'].to(device)
            # the input to each layer should be of shape (seq_len, batch_size, hidden_size)
            x = x.permute(1, 2, 0, 3)

            source_loader = data['source_loader']

            output = []
            output.append(merged_model(x[0]))
            for task_model in task_models:
                output.append(task_model(x[1]))

            # output[i]: (seq_len, batch_size, num_features) -> (batch_size, seq_len, num_features)
            for i in range(len(output)):
                output[i] = output[i].permute(1, 0, 2).cpu()

            # each datapoint consists of (torch.Tensor([output_merged, output_task_i]), source_loader)
            for i in range(output[0].size(0)):
                data_list = []
                data_list.append(output[0][i])
                data_list.append(output[source_loader[i]+1][i])
                data_tensor = torch.stack(data_list)
                transformed_data.append((data_tensor, source_loader[i]))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=data_loader.batch_size,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader
