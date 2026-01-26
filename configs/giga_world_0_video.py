config = dict(
    runners=['giga_world_0.GigaWorld0Trainer'],
    project_dir='/raid/vemburaj.yadav/giga-world-0/experiments/giga_world_0_video/it2w/fr3_single_franka_hand/ft_experiment_4',
    launch=dict(
        gpu_ids=[3],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                '/raid/vemburaj.yadav/giga-world-0/packed_data/fr3_single_arm_franka_hand/splits/train',
            ],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='GigaWorld0Transform',
                num_frames=93,
                height=480,
                width=640,
                fps=16,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        vae_model_path='/raid/vemburaj.yadav/.cache/huggingface/hub/vae',
        transformer_model_path='/raid/vemburaj.yadav/.cache/huggingface/hub/gigaworld0_video_pretrain_2b/transformer',
        train_mode='lora',
        lora_rank=64,
    ),
    # optimizers=dict(
    #     type='AdamW',
    #     lr=2 ** (-14.5),
    #     weight_decay=1e-2,
    # ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2e-4,
        # lr=2 ** (-14.5),
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_epochs=100,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',  # fp16, bf16, fp8
        checkpoint_interval=500,
        checkpoint_total_limit=100,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
    ),
)
