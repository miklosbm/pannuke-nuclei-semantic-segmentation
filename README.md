# Multi-class Nuclei Semantic Segmentation on the PanNuke Histopathology Dataset


This project is a step-by-step study of improving nuclei segmentation on the PanNuke dataset. Four models were built and trained in PyTorch, and after each one the weak  points (overfitting, limited architecture, no pretrained features) were analysed and addressed in the next. The test Dice improved from 0.49 to 0.65, or from 0.42 to 0.63 when averaged per image.

## Task

Classify every pixel of a 256×256 tissue image into one of six classes:
Background, Neoplastic, Inflammatory, Connective, Dead, Epithelial.

- **Data:** [PanNuke](https://huggingface.co/datasets/RationAI/PanNuke), Fold 1 only (2,656 image–mask pairs)
- **Split:** 70 % train / 15 % validation / 15 % test (seed 54, identical for all models)
- **Loss:** Dice loss
- **Metric:** mean Dice over the classes present in the image or the prediction (classes missing from both are skipped)

## Models

| # | Model | What changes | Batch | LR | Epochs |
|---|-------|--------------|-------|----|--------|
| 1 | SimpleUNet | baseline U-Net, no dropout, no augmentation | 64 | 1e-3 | 80 |
| 2 | SimpleUNet2 | + dropout (0.2) + augmentation | 64 | 5e-4 | 120 |
| 3 | UNet++ | dense skip connections, dropout, augmentation | 16 | 1e-4 | 160 |
| 4 | Pretrained UNet++ | ResNet34 encoder pretrained on ImageNet (`segmentation-models-pytorch`), AdamW | 16 | 1e-4 | 200 |

Augmentation: flips, 90°/180°/270° rotation, brightness / contrast / saturation (±20 %).

## Results

| Model | Best val Dice | Test Dice (per image) | Test Dice (per batch) |
|-------|--------------:|----------------------:|----------------------:|
| SimpleUNet | 0.4898 | 0.4159 | 0.4926 |
| SimpleUNet2 | 0.5262 | 0.4551 | 0.5311 |
| UNet++ | 0.5620 | 0.4806 | 0.5751 |
| **Pretrained UNet++** | **0.6584** | **0.6266** | **0.6488** |

*Per image:* Dice computed for every test image and averaged over the 398 test images (`05_DataEvaluation_summary.ipynb`).
*Per batch:* the value printed at the end of notebooks 01–04, averaged over batches (batch size 64 for models 1–2, 16 for models 3–4), which is more forgiving.
Both give the same ranking.

The result images (training curves and visual predictions) can be found in the `images` folder. A small model comparison on the same test images can be seen here:

<img width="1773" height="1872" alt="comparison" src="https://github.com/user-attachments/assets/2bffe1d2-4673-4a06-ada6-2efddcf6a1be" />

## Project structure

| File | Content |
|------|---------|
| `00_DataExploration.ipynb` | dataset exploration |
| `01_SimpleUNet.ipynb` … `04_PretrainedUNetPlusPlus.ipynb` | training and evaluation of each model |
| `05_DataEvaluation_summary.ipynb` | all four models on the same test images: side-by-side predictions and test Dice |
| `pannuke_dataset.py`, `pannuke_utils.py` | dataset class, Dice score / loss, training loop, plotting |
| `model_*.py` | model definitions |

## How to run

1. Install the dependencies (`uv sync`, or `pip install -r` equivalent from `pyproject.toml`).
2. Download PanNuke Fold 1 and put `images.npy` and `masks.npy` into `data_root/`.
3. Run the notebooks `01` → `04`. Each one trains a model and saves its best checkpoint.
4. Run `05` to compare the models.

Trained checkpoints are not included in the repository (the pretrained one is ~300 MB).

## Limitations and next steps

- Only Fold 1 was used; Folds 2 and 3 would give more than 7,000 training patches.
- Rare classes (Dead, Inflammatory) are often missed or confused; class weighting or stronger augmentation could help.
- More advanced architectures (Attention U-Net, HoVer-Net, CellViT) reach higher scores in the literature.

## References

If you use the PanNuke dataset, please cite:

```bibtex
@article{gamper2020pannuke,
  title={PanNuke Dataset Extension, Insights and Baselines},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Graham, Simon and Jahanifar, Mostafa and Khurram, Syed Ali and Azam, Ayesha and Hewitt, Katherine and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2003.10778},
  year={2020}
}
```

```bibtex
@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benes, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
  organization={Springer}
}
```
