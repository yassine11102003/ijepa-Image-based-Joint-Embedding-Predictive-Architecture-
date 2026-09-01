# Encoder (context + target) and predictor architecture
MODEL_CONFIG = {
    "img_size":            32,
    "patch_size":          4,
    "in_channels":         3,
    "embed_dim":           128,
    "depth":               6,
    "num_heads":           4,
    "mlp_ratio":           4,
    "predictor_dim":       64,
    "predictor_depth":     4,
    "predictor_num_heads": 4,
}

# Block-masking (context/target sampling)
MASK_CONFIG = {
    "num_target_blocks":          4,
    "target_scale_range":         (0.15, 0.2),
    "target_aspect_ratio_range":  (0.75, 1.5),
    "context_scale_range":        (0.85, 1.0),
    "context_aspect_ratio_range": (1.0, 1.0),
}

# Training
TRAIN_CONFIG = {
    "batch_size":   64,
    "epochs":       20,
    "lr":           1e-3,
    "ema_momentum": 0.996,
}

# Linear probe evaluation (frozen encoder + trainable linear classifier)
PROBE_CONFIG = {
    "batch_size":  128,
    "epochs":      10,
    "lr":          1e-3,
    "num_classes": 10,
}
