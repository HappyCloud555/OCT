""" This file builds model for the framework """
import logging
import copy
from model import wideresnet, resnet

models = {
    "wideresnet": wideresnet.build,
    "resnet50": resnet.resnet50
}


def build(cfg, logger=None, is_temp=False):
    # init params
    init_params = copy.deepcopy(cfg)
    type_name = init_params.pop("type")
    if type_name == "wideresnet":
        init_params["is_temp"] = is_temp
        if not is_temp:
            logger.info("================="+type_name+" init_params===============")
            logger.info(init_params)
    # init model
    model = models[type_name](**init_params)
    if not is_temp:
        if logger is not None:
            logger.info("{} Total params: {:.2f}M".format(
                type_name, sum(p.numel() for p in model.parameters()) / 1e6))
        else:
            logging.info("{} Total params: {:.2f}M".format(
                type_name, sum(p.numel() for p in model.parameters()) / 1e6))
    return model
