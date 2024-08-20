task_conf = dict(
    hist_len=96,
    pred_len=720,

    batch_size=64,
    max_epochs=10,
    lr=0.0001,
    optimizer="AdamW",
    optimizer_betas=(0.95, 0.9),
    optimizer_weight_decay=1e-5,
    lr_scheduler='StepLR',
    lr_step_size=1,
    lr_gamma=0.5,
    gradient_clip_val=5,
    val_metric="val/loss",
    test_metric="test/mae",
    es_patience=3,

    num_workers=2,
)