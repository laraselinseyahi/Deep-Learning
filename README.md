# Diabetic Retinopathy Classification using Pre-trained Deep Learning Models

This project involves fine-tuning pre-trained deep learning models—ResNet-50, InceptionNet, and AlexNet—for the binary classification of Diabetic Retinopathy. The implementation is carried out in a Colab notebook, leveraging the Kaggle dataset for training and evaluation.

## Dataset
The dataset used for this project can be found here: 
https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data
Pre-trained models used are from the ImageNet dataset.

Description: The dataset consists of labeled fundus images of the retina, which are classified into two categories:
Class 0: No Diabetic Retinopathy
Class 1: Diabetic Retinopathy

## Objective
The goal of this project is to fine-tune pre-trained Convolutional Neural Networks (CNNs) for accurate binary classification of Diabetic Retinopathy. By leveraging transfer learning, the project aims to achieve high performance while minimizing computational resources and training time.

## Key Features
Transfer Learning

Fine-tuning ResNet-50, InceptionNet, and AlexNet pre-trained on ImageNet.
Freezing lower layers to retain general features and fine-tuning upper layers for domain-specific learning.
Data Preprocessing

Resizing fundus images to the required input dimensions for each architecture.
Normalization for faster convergence.
Data augmentation (e.g., rotation, flipping) to improve model robustness.
Training and Validation

Binary Cross entropy loss function for classification.
Use of Adam optimizer with a learning rate scheduler.
Metrics: Accuracy, Precision, Recall, and F1 Score.

## How to Run the Project
Clone the repository and upload it to Google Colab.
Download the dataset from Kaggle and upload it to your Google Drive.
Update the file paths in the notebook accordingly.
Install dependencies:
!pip install tensorflow keras matplotlib seaborn scikit-learn  
Execute the cells in the Colab notebook to preprocess the data, fine-tune the models, and evaluate the results.


