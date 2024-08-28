# Teeth Classification with ResNet50

This project implements a Convolutional Neural Network (CNN) using ResNet50 to classify images of teeth into seven different categories. The model is built and trained using TensorFlow and Keras.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project consists of images categorized into seven classes:

- CaS (Caries)
- CoS (Composite Restoration)
- Gum
- MC (Metal Crown)
- OC (Orthodontic Case)
- OLP (Oral Lichen Planus)
- OT (Other Teeth Anomalies)

The data is split into three directories:
- `Training`: Used for model training.
- `Validation`: Used for validation during training.
- `Testing`: Used for final model evaluation.

## Model Architecture

The model is based on the ResNet50 architecture, pre-trained on ImageNet. The top layers are removed and replaced with:

- Global Average Pooling layer
- Dense layer with 1024 units and ReLU activation
- Dense layer with 512 units and ReLU activation
- Dense layer with 64 units and ReLU activation
- Dropout layer with a rate of 0.5
- Output layer with 7 units and softmax activation for multi-class classification

## Training Process

The model is trained with the following settings:

- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32

### Data Augmentation

Data augmentation is applied to the training dataset to improve generalization. The augmentation techniques include:

- Rescaling
- Rotation (up to 30 degrees)
- Width and Height Shifts (up to 20%)
- Shear Transformations
- Zoom (up to 20%)
- Horizontal Flips

### Callbacks

- **ReduceLROnPlateau**: Reduces the learning rate when validation loss plateaus.
- **ModelCheckpoint**: Saves the best model based on validation accuracy.

## Evaluation

The model is evaluated on both the validation and test datasets. The following metrics are used for evaluation:

- **Loss**
- **Accuracy**

A confusion matrix is also generated to visualize the performance of the model across different classes.

## Results

- **Validation Accuracy**: ~93.8%
- **Test Accuracy**: ~96.3%

These results indicate that the model performs well on the unseen test data, suggesting good generalization.

## Installation

To run this project locally, you need to have Python and the required libraries installed.

1. Clone this repository:
   ```code
   git clone https://github.com/abou-zithar/teeth-classification-resnet50.git
  

## Acknowledgments
This project was developed as part of a computer vision project related to a Cellula internship. Special thanks to the contributors and the open-source community for providing resources and tools.
