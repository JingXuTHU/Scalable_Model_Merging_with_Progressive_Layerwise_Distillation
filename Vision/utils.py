import torch
from merge_models import MergedModel

dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


def create_preresblock_activation_extractor(model):
    def extract__preresblock_activation(x):
        class StopForwardPass(Exception):
            pass

        resblock_input = []

        def create_hook():
            def hook(module, input, output):
                resblock_input.append(input[0])
                raise StopForwardPass
            return hook

        block = model.model.visual.transformer.resblocks[0]
        handle = block.register_forward_hook(create_hook())

        try:
            model(x)
        except StopForwardPass:
            pass

        handle.remove()

        return resblock_input[0]

    return extract__preresblock_activation

def get_avg_merged_model(args, merge_coef=0.3):
    pretrained_checkpoint = f'{args.save}/zeroshot.pt'

    image_encoder_pretrained = torch.load(
        pretrained_checkpoint, map_location=args.device)
    new_state_dict = image_encoder_pretrained.state_dict()

    for dataset in dataset_list:
        image_encoder = torch.load(
            f'{args.save}/{dataset}/finetuned.pt', map_location=args.device)
        for name, param in image_encoder_pretrained.named_parameters():
            new_param = (dict(image_encoder.named_parameters())[
                         name]-dict(image_encoder_pretrained.named_parameters())[name]) * merge_coef
            new_state_dict[name] = new_state_dict[name] + new_param
        del image_encoder
        torch.cuda.empty_cache()

    image_encoder_pretrained.load_state_dict(new_state_dict)
    return image_encoder_pretrained


def load_merged_layers(args, layer_idx, granularity='elementwise'):
    pretrained_checkpoint = f'{args.save}/zeroshot.pt'
    image_encoder_pretrained = torch.load(pretrained_checkpoint, map_location=args.device)
    layer_pretrained = image_encoder_pretrained.model.visual.transformer.resblocks[layer_idx]

    layers = []
    for dataset in dataset_list:
        image_encoder = torch.load(
            f'{args.save}/{dataset}/finetuned.pt', map_location=args.device)
        layer = image_encoder.model.visual.transformer.resblocks[layer_idx]
        layers.append(layer)

        del image_encoder
        torch.cuda.empty_cache()

    merged_layers = MergedModel(layer_pretrained, layers, granularity=granularity)

    del image_encoder_pretrained
    torch.cuda.empty_cache()

    return merged_layers, layers

def load_pretrained_layer(args, layer_idx):
    pretrained_checkpoint = f'{args.save}/zeroshot.pt'
    image_encoder_pretrained = torch.load(pretrained_checkpoint, map_location=args.device)
    layer_pretrained = image_encoder_pretrained.model.visual.transformer.resblocks[layer_idx]

    return layer_pretrained


def load_image_encoders(args, dataset_list=['MNIST', 'EuroSAT', 'GTSRB', 'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']):
    image_encoders_arr = []
    for dataset in dataset_list:
        image_encoder = torch.load(f'{args.save}/{dataset}/finetuned.pt', map_location=args.device)
        image_encoders_arr.append(image_encoder)

    return image_encoders_arr
