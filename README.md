# Iris - Ocular Disease Recognition

This project aims to develop an artificial intelligence solution for the automatic recognition of eye diseases from medical images.  
My main aim is to help advance the use of AI in medicine. Feel free to modify and republish this CNN model !  

## Usage
The data used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).  
Unzip the zip file "archive" and put it at the root of the project.   
Run the Iris.py file and be patient. You can adapt the code to use the model on specific images.  

## Information
The model is based on the VGG19 architecture, previously trained on the vast ImageNet dataset, offering expertise in image recognition. As part of this project, the following steps are being taken to adapt this architecture to the specific task of classifying eye diseases:  
  
Weights already trained on ImageNet are used to initiate VGG19. In order to preserve the knowledge acquired, the parameters of all VGG19 layers are frozen.  
Additional layers are superimposed on top of VGG19 to adapt the model to the classification of ocular diseases.  
A final Dense layer is added, with softmax activation, to predict the output class from 8 potential categories (normal, cataract, diabetes, etc.)  
To avoid overlearning, a dropout regularisation technique is applied, helping the model to generalise efficiently to new data.
