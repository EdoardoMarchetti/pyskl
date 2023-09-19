modality = 'j'
graph = 'gymnasio'
work_dir = f'./work_dirs/dgstgcn/ntu60_xsub_3dkp/fine_tuning/youtube/{modality}'

model = dict(
    cls_head=dict(type='GCNHead', num_classes=20, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/youtube_pwl.pkl'
mapping_file = None

train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=10),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, mapping_file = mapping_file,  pipeline=train_pipeline, split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file, mapping_file = mapping_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, mapping_file = mapping_file, pipeline=test_pipeline, split='xsub_val'))

load_from = INSERIRE IL PERCORSO AL MODELLO

# optimizer, 4GPU
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
