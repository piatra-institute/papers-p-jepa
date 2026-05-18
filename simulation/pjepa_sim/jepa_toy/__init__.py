"""NumPy JEPA toy on dishworld for testing JEPA augmentations.

This module exists to give *directional* evidence about which auxiliary
losses are worth trying at V-JEPA scale. The toy is deliberately small:
- 11-dim contexts (sensor + visual features) from dishworld
- 2-layer MLP encoder, EMA target encoder, 2-layer predictor
- 4 hidden regimes, 4 direct actions, deterministic seeds

A toy positive result here is not a real V-JEPA result. A toy negative
result is a strong signal that the augmentation will likely not pay at
scale and should be reordered down the priority list.

See docs/JEPA_AUGMENTATIONS.md for the PyTorch / V-JEPA-scale
specifications, and experiments/h5_jepa_augmentations.py for the
ablation runner.
"""
