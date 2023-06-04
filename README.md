# Ensemble Learning for Multi-Class Classification

This repository contains a Python script for implementing ensemble learning techniques to perform multi-class classification. The script uses PyTorch and torchvision libraries for model training and evaluation.

<!-- TOC -->
* [Ensemble Learning for Multi-Class Classification](#ensemble-learning-for-multi-class-classification)
<!-- TOC -->

## Dataset
The dataset used for this script consists of images and additional data associated with each image. The ImageDataWithAdditionalDataset class is a custom dataset class that loads the images and additional data from separate folders. It uses the datasets.ImageFolder class from torchvision to load the images and associates the additional data with each image. The dataset is then split into training, validation, and test sets using the random_split function from PyTorch

## Models
The script utilizes three pre-trained models for ensemble learning: ResNet50, DenseNet121, and VGG16. Each model is adapted for multi-class classification by modifying the output layer. The script freezes the parameters of certain layers to perform fine-tuning only on the last fully connected layer. The models are trained using the Adam optimizer and VariableCostLoss as the loss function.

## Techniques for Learning Improvement
The script incorporates two techniques to improve the learning process:
- Data Augmentation: Data augmentation is applied to the training data using random transformations such as random resized crop, random horizontal flip, color jitter, and random rotation. This technique helps to increase the diversity of the training data and improves the model's generalization capability.
- Learning Rate Scheduling: The learning rate of the optimizer is adjusted using the StepLR scheduler from PyTorch. The learning rate is reduced by a factor of 0.1 after a certain number of epochs. This technique helps to fine-tune the learning process and achieve better convergence.

## MLflow Integration
The script integrates MLflow, an open-source platform for managing the machine learning lifecycle. It starts an MLflow experiment and logs relevant metrics and parameters during the training process. The training progress, including training loss and validation accuracy, is visualized using matplotlib. The training progress plot and the confusion matrix are saved as artifacts in the MLflow experiment for further analysis.

## Results
The script trains the ensemble of models for a specified number of epochs. It prints the training losses and validation accuracy at each epoch. The training progress, including the training loss and validation accuracy, is displayed as plots. The confusion matrix is also displayed, representing the normalized confusion matrix for the validation set.

All the relevant metrics, parameters, and artifacts are logged in the MLflow experiment, which can be accessed and analyzed through the MLflow UI.