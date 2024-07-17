[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14998012&assignment_repo_type=AssignmentRepo)
# XNAP-Project Business Classification Using Neural Networks
## Introduction
Welcome to the Business Classification project. This repository contains the implementation of a multi-modal neural network designed to classify various types of businesses based on street view images. The goal is to distinguish between 28 different business categories, such as bakery, pet shop, restaurant, etc.

## Objective
The main objective of this project is to develop a model capable of classifying various types of businesses based on street view images. To enhance classification accuracy, the model utilizes both visual features and textual information extracted from the images.

## Dataset
The data used in this project comes from the Con-Text dataset which includes a wide variety of images of different business types captured from the street, providing a rich and diverse representation.

The Con-Text dataset is constructed from sub-categories of the ImageNet “building” and “place of business” sets to evaluate fine-grained classification. It consists of 28 categories with a total of 24,255 images. This dataset is not specifically designed for text recognition, so not all images contain text. Additionally, the high variability in text size, location, resolution, and style, combined with uncontrolled environmental settings (such as illumination), makes text recognition in this dataset more challenging.

Additionally, since some images contain text that can be decisive for classification, we incorporated OCR (Optical Character Recognition) tags. These tags provide the textual information present in the images, which is crucial for the accurate identification of certain business types often recognized by visible names or signs.

![unnamed (8)](https://github.com/DCC-UAB/xnap-project-matcad_grup_3-1/assets/92937869/b361a937-1c79-4fd7-9e48-44a64c4ecb9f)

As we can see there are some classes with less images in the dataset than others. Moreover, most of these classes are the ones with a worse prediction. So, uploading new pictures of these classes to the dataset would be a great practice.

## Model Architecture
Our model is a multimodal model whose objective is to extract visual and textual information from the image in order to make a classification. These types of models are known as Multimodal Models. These two parts then feed into a Transformer that ultimately classifies the images.

### Visual Part
We started with a base model using a ResNet-50 with all parameters frozen, meaning it functions as a feature extractor, which means that The ResNet-50 extracts relevant features from the images.

### Textual Part
For the textual part of our model, we used a combination of OCR and FastText to process and analyze the text present in the images. The OCR extracts textual information such as business names, signs, and other relevant texts and converts them into digital data. This text vector is then passed to FastText, which transforms the words into numerical vectors representing their semantics.

### Transformer
The extracted visual features from ResNet-50 and the text vectors from OCR and FastText are passed through linear projections to adjust the dimensions, preparing them to be fed into the Transformer Encoder. The Transformer combines and processes these multimodal representations.

An important component in this architecture is the classification token (CLS). The CLS token is concatenated at the beginning of the input sequence to the Transformer and is used to aggregate the information from the entire sequence. During training, the Transformer learns to use the CLS token information for the final prediction. The resulting vector of the CLS token at the end of the Transformer contains a combined representation of all visual and textual features the model has processed.

Finally, the CLS token output is passed through an MLP (Multilayer Perceptron) head layer that makes the final classification of the image into one of the predefined classes.

## Original Model and Results
![unnamed (6)](https://github.com/DCC-UAB/xnap-project-matcad_grup_3-1/assets/92937869/98b8679e-7f0a-4412-9ff7-c7940da6f0dc)
![unnamed (7)](https://github.com/DCC-UAB/xnap-project-matcad_grup_3-1/assets/92937869/8e40188a-4266-4c46-9099-cdd89349231d)

In the presented graphs, we can observe the original model’s performance during the training process, measured through accuracy and validation loss over different training steps.

### Accuracy Graph
The first graph shows the model’s accuracy during training. Initially, the model starts improving rapidly in the early stages. As training continues, the accuracy growth moderates but maintains an upward trend, stabilizing around 77%, indicating consistent performance.

### Validation Loss Graph
The second graph shows the model's validation loss, starting high at around 1.3. It decreases rapidly to about 0.9 in early stages, then more gradually, stabilizing around 0.8. Small fluctuations are normal, but overall, the loss remains low and consistent.

## Improvements to the Model
- ResNet-152: we replaced ResNet-50 with ResNet-152, a deeper model that captures more detailed features, which is crucial for complex tasks. Initially, all parameters of ResNet-152 are frozen except for the last three layers, allowing for fine-tuning during training.
- Optimizer: we used AdamW, which helps improve model generalization and prevents overfitting.
- Data Augmentation: applied techniques like rotation, resizing, and color adjustment to enhance the training dataset.
- Learning Rate Scheduler: MultiStepLR adjusts the learning rate during training for efficient convergence.

## Performance Analysis
The improved model showed a nearly 4% increase in accuracy (81,2%) compared to the original model. As epochs progress, the loss of the original model tends to stabilize around 0.8, while the improved model stabilizes around 0.65.

If we had more time and computational resources, we could further enhance the model by exploring advanced fine-tuning and data augmentation techniques and incorporating additional data.

![Captura de pantalla 2024-05-31 170442](https://github.com/DCC-UAB/xnap-project-matcad_grup_3-1/assets/92937869/3ecec98c-6b32-4041-bda8-4c31ef983e6c)

As we can observe from the images, there are some similar classes which the model does not predict well. Some of them are really close to each other, such as “Bistro”, “TeaHouse” or “Tavern”. To enhance the performance of the model, some classes could be grouped and named as “Restaurant”.

## Repository Structure
- .github/: GitHub-related configurations and feedback.
- models/: Contains the model architectures and weights.
- splits/: Contains 3 splits of train and test data. 
- utils/: Utility scripts and helper functions.
- .gitignore: Git ignore file to exclude unnecessary files.
- ConTextTransformer_inference.ipynb: Jupyter notebook for model inference.
- LICENSE: License information for the project.
- README.md: Project overview and documentation.
- environment.yml: Environment configuration file for dependencies.
- inference_plots.ipynb: Jupyter notebook for analysis and visualization of inference results.
- main.py: Main script for running the final model.
- splits_analysis.ipynb: Compares the final model's accuracy between each split(subset of images of each class). It is also shown the distribution of images of each class in each split.
- test.py: Script for testing the model structure and functionality.
- train.py: Script for training the model.

## Example Code
Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-project
```

To run the example code:
```
python main.py
```



## Contributors
Manel Carrillo Maíllo (1633426@uab.cat), Nil Farrés Soler (1635864@uab.cat), Alba Puig Font (1636034@uab.cat), Genís Ruiz Menárguez (1633623@uab.cat)

Xarxes Neuronals i Aprenentatge Profund,

Computational Mathematics & Data Analytics, 

UAB, 2024
