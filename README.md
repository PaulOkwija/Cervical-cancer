# Cervical Cancer Lesion Segmentation

This repository provides code and resources for lesion segmentation in cervical cancer screening using colposcopy images. It includes data preparation utilities, multiple deep learning model architectures, custom loss functions, and evaluation metrics, as well as exploratory and analysis notebooks.

## Directory Structure
```
Cervical-cancer/
│
├── 0.5_agreement_extraction.py         # Script for agreement analysis between datasets
├── metrics_and_losses.py               # Custom metrics and loss functions
├── README.md                           # Project documentation
├── model_definitions/                  # Model architectures and loss definitions
│   ├── losses.py                       # Custom loss functions
│   ├── mobilenet.py                    # MobileNet-based UNet model
│   ├── UNet.py                         # UNet model implementation
│   ├── vgg16_encoder.py                # VGG16-based UNet model
├── notebooks/                          # Jupyter notebooks for EDA and analysis
│   ├── csv_analysis.ipynb              # CSV data analysis
│   ├── data_collection.ipynb           # Data collection and JSON structure analysis
│   ├── EDA.ipynb                       # Exploratory Data Analysis
│   ├── lesion_segmentation.ipynb       # End-to-end workflow for cervical lesion segmentation
├── saved_models/                       # Directory for saving trained models
├── utils/                              # Utility scripts
│   ├── data_preparation.py             # Data preprocessing and augmentation
│   ├── utils.py                        # General utilities (image reading, metrics, etc.)
```

## Data Preparation
- Utilities for reading, preprocessing, and augmenting images and masks are in `utils/data_preparation.py` and `utils/utils.py`.
- Data can be loaded from CSV or JSON files, with functions for train/test splitting and augmentation (flip, rotate, etc.).

## Model Architectures
- **UNet**: Standard UNet implementation in `model_definitions/UNet.py`.
- **VGG16-UNet**: UNet with VGG16 encoder in `model_definitions/vgg16_encoder.py`.
- **MobileNet-UNet**: Lightweight UNet using MobileNetV2 as encoder in `model_definitions/mobilenet.py`.

## Training & Evaluation
- Custom loss functions and metrics are defined in `model_definitions/losses.py` and `metrics_and_losses.py` (Dice, IoU, Tversky, etc.).
- The repository supports training with different architectures and evaluating segmentation performance.

## Utilities
- `utils/utils.py` provides helper functions for image reading, dataset preparation, and metric calculation (IOU, Dice, etc.).
- `0.5_agreement_extraction.py` analyzes agreement between different reviewers' segmentations.

## Notebooks
- **EDA.ipynb**: Data exploration and visualization.
- **data_collection.ipynb**: Analysis of JSON data structure and content.
- **csv_analysis.ipynb**: Analysis of CSV data from reviewers.
- **lesion_segmentation.ipynb**: End-to-end workflow for cervical lesion segmentation, including data preparation, model training, evaluation, and visualization of predictions.

<!-- ## How to Run
1. Prepare your dataset in the required format (see notebooks for examples).
2. Use the data preparation utilities to preprocess and split your data.
3. Select a model architecture from `model_definitions/`.
4. Train and evaluate your model using the provided scripts and metrics.
5. Use the notebooks for data analysis and visualization. -->

## Requirements
- Python 3.x
- TensorFlow, Keras, scikit-learn, pandas, numpy, opencv-python, Pillow, matplotlib, tqdm


## References
- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [VGG16: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

---
For questions or contributions, please open an issue or pull request.
