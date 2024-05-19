import copy
import os

import torchvision
from torchvision import datasets, transforms
from dataset.randaugment import RandAugmentMC
other_func = {"RandAugmentMC": RandAugmentMC}


def get_trans(trans_cfg):
    init_params = copy.deepcopy(trans_cfg)
    type_name = init_params.pop("type")
    if type_name in other_func.keys():
        return other_func[type_name](**init_params)
    if type_name == "RandomApply":
        r_trans = []
        trans_list = init_params.pop('transforms')
        for trans_cfg in trans_list:
            r_trans.append(get_trans(trans_cfg))
        return transforms.RandomApply(r_trans, **init_params)
    elif hasattr(transforms, type_name):
        return getattr(transforms, type_name)(**init_params)
    else:
        raise NotImplementedError(
            "Transform {} is unimplemented".format(trans_cfg))


class BaseTransform(object):
    """ For torch transform or self write
    """
    def __init__(self, pipeline):
        """ transforms for data

        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipeline = pipeline
        self.transform = self.init_trans(pipeline)

    def init_trans(self, trans_list):
        trans_funcs = []
        for trans_cfg in trans_list:
            trans_funcs.append(get_trans(trans_cfg))
        return transforms.Compose(trans_funcs)

    def __call__(self, data):
        return self.transform(data)


class ListTransform(BaseTransform):
    """ For torch transform or self write
    """
    def __init__(self, pipelines):
        """ transforms for data

        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipelines = pipelines
        self.transforms = []
        for trans_dict in self.pipelines:
            self.transforms.append(self.init_trans(trans_dict))

    def __call__(self, data):
        results = []
        for trans in self.transforms:
            results.append(trans(data))
        return results


def build(cfg):
    # build dataset , give boe dataset example here
    if cfg.type == "OCT":
        transform_ulabeled = ListTransform(cfg.upipelinse)
        train_unlabeled_dataset = datasets.ImageFolder(os.path.join(cfg.root, "unlabel"),  transform=transform_ulabeled)
        if cfg.dataset == "BOE":
            transform_labeled = ListTransform(cfg.lpipelines)
            train_labeled_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.root, "train"), transform=transform_labeled)
        transform_test = BaseTransform(cfg.tpipeline)
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.root, "test"), transform=transform_test)
        transform_val = BaseTransform(cfg.vpipeline)
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.root, "val"), transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset