# Set the number of classes here
num_classes = 3
train = dict(eval_step=1024,
             total_steps=1024*4,
             trainer=dict(type="FixMatchCCSSL",
                          threshold=0.95,
                          T=1.,
                          temperature=0.07,
                          lambda_u=1.,
                          lambda_contrast=1.,
                          contrast_with_softlabel=True,
                          contrast_left_out=True,
                          contrast_with_thresh=0.8,
                          loss_x=dict(
                              type="cross_entropy",
                              reduction="mean"),
                          loss_u=dict(
                              type="cross_entropy",
                              reduction="none"),
             )
)
model = dict(
     type="wideresnet",
     depth=28,
     widen_factor=2,
     dropout=0.2,
     num_classes=num_classes,
     proj=True
)

data = dict(
    # OCT dataset
    type="OCT",
    num_workers=4,
    num_classes=num_classes,
    batch_size=8,
    expand_labels=False,
    mu=2,

    # DATASET setting
    dataset="BOE",
    # dataset root
    root="dataset/Semi_BOEdata",
    # numbers of unlabeled data
    num_labeled=55,

    lpipelines=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="ToTensor")
    ]],

    upipelinse=[[
        # w
        dict(type="RandomHorizontalFlip"),
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="ToTensor")
        ],
        # s1
        [
        dict(type="RandomHorizontalFlip"),
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="RandAugmentMC", n=2, m=10),
        dict(type="ToTensor")
        ],
        # s2
        [
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomApply",
                transforms=[
                    dict(type="ColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1),
                ],
                p=0.8),
        dict(type="RandomGrayscale", p=0.2),
        dict(type="ToTensor")
        ]
    ],

    vpipeline=[
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="ToTensor")
    ],

    tpipeline=[
        dict(type="Resize",
             size=256),
        dict(type="CenterCrop",
             size=224),
        dict(type="ToTensor")
    ]
)

scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=train['total_steps']
)

ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
#apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html
amp = dict(use=False, opt_level="O1")

log = dict(interval=1)
ckpt = dict(interval=1)

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.001, nesterov=True)
