# Diabetic Retinopathy Classification using Pre-trained Deep Learning Models

This project explores fine-tuning pre-trained deep learning models—ResNet-50, InceptionNet, and AlexNet—for the binary classification and multiclass classification tasks of Diabetic Retinopathy. The implementations are carried out in a Colab notebook.

## Dataset
# Binary Classification Data
I used a publicly available dataset obtained from Kaggle, titled “Diagnosis of Diabetic Retinopathy”
https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data
The dataset contains a collection of retinal images with labels indicating the presence (1) or absence (0) of diabetic
retinopathy, making it suitable for binary classification. The images are high-resolution retinal scans assessed by
medical professionals. The dataset consists of 1050 DR and 1026 non-DR images for training, 245 DR and 286 non-DR
images for validation, and 113 DR and 118 non-DR images for testing.
Figure 1: DR positive Figure 2: DR negative
Figure 3: Binary Classification Images: DR positive and DR negative
# Multiclass Classification Data
A publicly available dataset from Kaggle titled “Diabetic Retinopathy Dataset” for the multiclass clas-
sification problem. The dataset includes pre-processed retinal fundus images of size 256 x 256.
https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset/data The dataset includes a total of
2750 retinal images in 5 categories with the corresponding labels and number of images per each category:
Class 0: Healthy (Not DR) – 1000
Class 1: Mild DR – 370
Class 2: Moderate DR – 900
Class 3: Proliferative DR – 290
Class 4: Severe DR – 190
Using the train_test_split function and 70/20/10 split, Train set had 1974, Validation set had 566, and Test set
had 280 images.
For both tasks, during data pre processing, the images were resized according to the input requirements of the pre
trained model they got trained with: (224, 224, 3) for Resnet 50 and Alexnet, and (299, 299, 3) for InceptionNet.
Normalization was performed using the preprocess_input function in Keras for each specific model.
In multiclass classification, class imbalance was addressed by applying targeted data augmentation to underrepresented
classes (1, 3, and 4). I applied the following data augmentation techniques: 1) Random horizontal and vertical flips with
50% probability, 2) Random brightness adjustments with maximum delta = 0.2, 3) Random contrast adjustments with
range 0.8 to 1.2, 4) Random cropping and resizing to introduce diversity in spatial orientation. In binary classification,
random horizontal flips and random rotations were applied to enhance variability in the training set.

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


