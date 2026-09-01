# I-JEPA From Scratch

Implémentation from scratch (PyTorch) de **I-JEPA** — *Self-Supervised Learning
from Images with a Joint-Embedding Predictive Architecture* (Assran et al.,
Meta AI, CVPR 2023) — entraînée sur CIFAR-10 et évaluée par linear probing.

Contrairement à un autoencoder classique, I-JEPA n'apprend pas à reconstruire
des pixels : il apprend à prédire, dans l'**espace des représentations**, ce
qui se trouve dans des blocs masqués de l'image, à partir du contexte visible.

---

## Structure du projet

```
ijepa-from-scratch/
├── config.py          # Configs modèle, masquage, entraînement, linear probe
├── model.py           # PatchEmbed, MultiHeadAttention, Block, Encoder, Predictor
├── masking.py         # Échantillonnage des blocs contexte/cible (block-masking)
├── dataset.py          # CIFAR-10 (pretrain train/val split + probe train/test)
├── train_jepa.py       # Boucle d'entraînement JEPA (EMA, stop-gradient)
├── linear_probe.py     # Évaluation : encodeur gelé + classifieur linéaire
└── requirements.txt
```

---

## Le mécanisme, en bref

```
Image → PatchEmbed → grille de patches
                     ├── Target encoder (image complète) → représentations cibles
                     │       (mis à jour uniquement par EMA, jamais par backprop)
                     └── Context encoder (patches visibles seulement) → tokens de contexte
                                     │
                                     ▼
                                Predictor (mask token + positional embedding)
                                     │
                                     ▼
                        prédiction des représentations des blocs cibles
                                     │
                    Loss (Smooth L1) contre les représentations du target encoder
                         (uniquement sur les patches des blocs cibles)
```

- **Masquage par blocs** : 4 blocs cibles (échelle 0.15-0.2 de l'image, peuvent se chevaucher entre eux) + 1 grand bloc de contexte (échelle 0.85-1.0) dont on retire tous les patches appartenant à un bloc cible — empêche le predictor de "tricher" par interpolation locale.
- **Target encoder** : appliqué une seule fois sur l'image entière (self-attention globale), jamais entraîné par backprop — mis à jour par moyenne mobile exponentielle (EMA) des poids du context encoder, avec `stop-gradient`. C'est cette combinaison qui empêche l'effondrement des représentations (collapse), sans besoin de terme de régularisation explicite (contrairement à VICReg).
- **Predictor** : transformer plus étroit que l'encodeur (bottleneck), reçoit les tokens de contexte + un mask token partagé additionné du positional embedding de chaque position cible à prédire.
- **Loss** : Smooth L1 entre les prédictions et les représentations cibles, calculée uniquement sur les patches des blocs cibles.

---

## Simplifications par rapport au papier original

Ce projet vise à démontrer la compréhension du mécanisme sur une échelle réaliste pour une machine personnelle (CPU) / Google Colab gratuit — pas à reproduire les résultats du papier (ViT-Huge, ImageNet complet, TPU/GPU en cluster). Différences assumées :

- **CIFAR-10 (32×32)** au lieu d'ImageNet (224×224) — grille de patches beaucoup plus petite (8×8 = 64 patches, patch_size=4) qu'un ViT standard (14×14).
- **ViT miniature** : `embed_dim=128`, 6 couches pour l'encodeur ; `predictor_dim=64`, 4 couches pour le predictor — très en dessous des tailles du papier.
- **Masque partagé pour tout le batch** (un seul masque échantillonné par step), plutôt qu'un masque différent par image.
- **Momentum EMA fixe** (0.996), plutôt qu'un scheduling progressif vers 1.0 au fil de l'entraînement.
- **4 blocs cibles traités comme un ensemble unique** (union dédupliquée des patches, un seul appel au predictor par step), plutôt que 4 appels séparés au predictor comme dans l'implémentation officielle.

---

## Installation

```bash
uv venv
uv pip install -r requirements.txt
```

L'entraînement est lent sur CPU pur — un run complet a été effectué sur **Google Colab (GPU T4 gratuit)** plutôt qu'en local.

---

## Utilisation

### Pré-entraînement JEPA

```bash
python train_jepa.py
```

Découpe le split train de CIFAR-10 en train/val (90/10), entraîne pendant 20 epochs, n'écrase les checkpoints (`context_encoder.pth`, `predictor.pth`) que lorsque la val loss s'améliore — pas de surapprentissage silencieux sur la tâche de pré-entraînement elle-même.

### Évaluation (linear probe)

```bash
python linear_probe.py            # encodeur pré-entraîné
python linear_probe.py --random   # baseline : mêmes poids, jamais entraînés
```

Gèle l'encodeur, moyenne les représentations de patches en un vecteur par image, entraîne une simple couche linéaire dessus avec les vrais labels CIFAR-10 (split test, jamais vu pendant le pré-entraînement).

---

## Résultats

Entraînement sur Google Colab (GPU T4), 20 epochs de pré-entraînement (meilleur checkpoint retenu à l'epoch 9, val loss = 0.1070), puis 10 epochs de linear probe :

| Encodeur | Accuracy (test CIFAR-10) |
|---|---|
| **JEPA pré-entraîné** | **51.4 %** |
| Baseline (poids aléatoires, même architecture) | 33.7 % |

Écart de **+17.7 points** entre l'encodeur pré-entraîné et la baseline aléatoire — preuve que le pré-entraînement self-supervised apprend des représentations effectivement utiles pour une tâche en aval, sans jamais avoir vu de label pendant le pré-entraînement.

La courbe train/val a aussi validé l'intérêt du split de validation : la val loss atteint son minimum à l'epoch 9 puis remonte légèrement (léger surapprentissage sur la tâche de pré-entraînement les epochs suivantes) — le mécanisme de sauvegarde conditionnelle a bien conservé le meilleur checkpoint plutôt que celui de la dernière epoch.

---

## Références

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023 ([arXiv:2301.08243](https://arxiv.org/abs/2301.08243))
- Implémentation officielle : [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)
