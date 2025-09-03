import sys
sys.path.append(".")
from argparse import ArgumentParser
from typing import Dict
from basicsr.archs import build_network
import torch
import yaml
from collections import OrderedDict


def load_weight(weight_path: str) -> Dict[str, torch.Tensor]:
    weight = torch.load(weight_path)
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module."):]
        pure_weight[key] = val

    return pure_weight

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

parser = ArgumentParser()
parser.add_argument("--model_config", type=str, default="options/train/train_StereoINR_HATNet-L.yml")
parser.add_argument("--weight", type=str, default="weights/ASteISR/ASteISR_X4.pth")
parser.add_argument("--output", type=str, default="./weights/StereoINR_deformv6.pth")
args = parser.parse_args()

with open(args.model_config, 'r') as f:
    opt =  yaml.load(f, Loader=ordered_yaml()[0])

model = build_network(opt['network_g'])

sd_weights = load_weight(args.weight)['params']
scratch_weights = model.state_dict()

init_weights = {}


for weight_name in scratch_weights.keys():
    # find target pretrained weights for this weight
    if weight_name in sd_weights:
        target_weight = sd_weights[weight_name]
        target_shape = target_weight.shape
        model_shape = scratch_weights[weight_name].shape
        # if pretrained weight has the same shape with model weight, we make a copy
        if model_shape == target_shape:
            init_weights[weight_name] = target_weight.clone()
        # else we copy pretrained weight with additional channels initialized to zero
        else:
            newly_added_channels = model_shape[-1] - target_shape[-1]
            L, _ = target_shape
            zero_weight = torch.zeros((L, newly_added_channels)).type_as(target_weight)
            init_weights[weight_name] = torch.cat((target_weight.clone(), zero_weight), dim=-1)
            print(f"add zero weight to {weight_name} in pretrained weights, newly added channels = {newly_added_channels}")
    else:
        init_weights[weight_name] = scratch_weights[weight_name].clone()
        print(f"These weights are newly added: {weight_name}")
save_dict = {}
model.load_state_dict(init_weights, strict=True)
save_dict['params'] = model.state_dict()
torch.save(save_dict, args.output)
print("Done.")
