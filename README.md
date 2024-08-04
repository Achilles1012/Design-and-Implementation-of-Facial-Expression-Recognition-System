
# Design and Implementation of Facial Expression Recognition System
#### Under Supervision of Dr. Ningthoujam Johny Singh, Assistant Professor, NIT Meghalaya.
This project a is made under the **Summer Internship Program at NIT Meghalaya.**

The primary objective of this project is to develop a robust and accurate Facial Expression Recognition (FER) system that can classify facial expressions into several distinct emotional categories: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger and Contempt. This project aims to create a system that is not only accurate but also reliable, ensuring consistent performance across various demographic groups and conditions.

 The system seeks to utilize state-of-the-art machine learning techniques, particularly Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), to enhance its capability to learn from complex data and generalize well to unseen examples.

## Requirements
##### Setup for the project
- tensorflow == 2.10.0
- pytorch == 2.3.1
- cuda/cuDNN enabled
- Tesla P100-PCIE : 16GB   

To install the dependencies

    pip install -r requirements.txt

## Data Collection
1. FERPlus

This database was originally utilized in the ICML 2013 Challenges in Representation Learning. It consisted of 28709 grayscale images in training set and a public and private test set with 3589 images each. Many of the images were mislabeled hence a newer version of FER2013 called FERPlus was released with an added class (Contempt).

Available at https://github.com/microsoft/FERPlus

2. (Expression-in-the-Wild) ExPW

The dataset has a total of 91,793 images which were downloaded using Google Image search. It has been divided into seven expression classes.

Available at https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html

## Preproccessing

1. We use a Convolutional Neural Networks based model for the objective on FERPlus.
Images are resized to **48×48**  dimension and **grayscaled** to lower parameter size of the model.

2. We preproccess the images of ExPW according to the Vision Transformer(pre-trained on ImageNet-21K) **vit_base_patch16_224_in21k** model requirements can be met using ViTImageProcessor.
- Resize images to 224x224 pixels.

- Rescale pixel values from [0, 255] to [0, 1] using a factor of approximately 0.004.

- Normalize images using a mean and standard deviation of [0.5, 0.5, 0.5] for each RGB channel.

- Use bilinear interpolation for resizing.

## Modeling

FERPlus

    run  CNN_basedFER2Plus.ipynb

ExPW
    
    run vit_base_patch16_224_in21k_(ADAM)_ExPW.ipynb

## Deployment
System Architecture

To run the application :

    python EmotionsLive.py
