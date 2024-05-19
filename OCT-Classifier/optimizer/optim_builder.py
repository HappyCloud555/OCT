import copy
import torch.optim as optim

optimizer_dict = {
    "SGD": optim.SGD
}


def build(cfg, model):
    optimizer_cfg = copy.deepcopy(cfg)
    optim_type = optimizer_cfg.pop("type")
    no_decay = optimizer_cfg.pop("no_decay", ['bias', 'bn'])
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)],
            'weight_decay': optimizer_cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_dict[optim_type](grouped_parameters, **optimizer_cfg)
    return optimizer
