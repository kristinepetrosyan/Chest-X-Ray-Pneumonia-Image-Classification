## Module 4 Final Project


## Table of Contents
* [General Info](#General-Info)
* [Technologies](#Technologies)
* [Data](#Data)
* [Methodology](#Methodology)
* [Final Models' Results](#Final-Models'-Results)
* [Recommendations](#Recommendations)
* [Future Work](#Future-Work)

### General Info 
In this project we will work on well known [Chest X-Ray Images (Pneumonia)] dataset from Kaggle. The goal of this project is to predict whether the X-Ray images are belong to a healthy person or a pneumonia patient by applying neural network models.

The main goal of to this project is to increase recall score for pneumonia images (Sensitivity) above 90% and recall score for normal images (Specificity) above 90%.


## Technologies
This project was created using the following languages and libraries. An environment with the correct versions of the following libraries will allow re-production and improvement on this project. 

* Python version: 3.6.9
* Matplotlib version: 3.0.3
* Seaborn version: 0.9.0
* Sklearn version: 0.20.3
* TensorFlow version: 1.14.0
* Keras version: 2.3.0

### Data

The data obtained from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) has 5860 training images, divided into 3 fold of train, validation and test. The data is also highly imbalanced that number of pneumonia images exceeds the number of normal images nearly 3 times.

After manual redistribution of the dataset into 3 folders, train set contains 70%, test and validation and test sets contains 15% of the data and balanced shares of normal and pneumonia images.

It is important to note that initial accuracy levels of the deep learning models visibly increased after balanced re-distibution of the images into train, validation and test folders. 

### Methodology

To observe accuracy and recall scores throughout the models, 7 models applied. 

- Basic neural network model with 2 layers
- Regulatized basic neural networks model with dropout
- Convolutional neural networks model
- Deep convolutional neural networks model
- Xception
- VGG3
- VGG5 

Recall, accuracy and f1 scores are used for evaluation metrics. As the data is highly imbalanced to increase performance of the last model data augmentation is also applied to the dataset.

As the models got complicated, it is observed accuracy, sensitivity and specificity scores increased throughout the models. Also, it is noticed that data augmentation lead to rise in both recall scores for each labels and increased model performance.

### Final Models' Results

It is observed that accuracy and recall scores for minority models increased as the models got complicated.

<img src="https://github.com/kristinepetrosyan/Mod4project/blob/master/Comparison.png">

Among all the trained models, initially the best resulted obtained from CNN model. 




The model predicted;

- 109 False positives,
- 125 True positives,
- 384 True negatives,
- 6 False negatives.

<img src="https://github.com/kristinepetrosyan/Mod4project/blob/master/CNN_matrix.png">

The CNN model by performing 98% recall score for Specificity partially achieved the set threshold. However, it is still underperforming in terms of Sensitivity threshold.

<img src="https://github.com/kristinepetrosyan/Mod4project/blob/master/CNN1.png">


### Recommendations

- The model should be used as a tool by medical experts and specialists which will support their diagnosis and treatment method.
- To reach higher levels of accuracy and recall score oversampling techniques should be used.
- Balancing the number of labels may also lead to higher accuracy and recall scores. 



### Future Work 
The Dataset can be enriched, and the target variables can be balanced by oversampling methods. Different pre-trained models can be applied to observe accuracy , f1, precision and recall scores. The model can be trained over to detect the cause of pneumonia (bacteria or COVID19). COVID19 dataset will be loaded as a test in order to automatically detect the virus. Conclusion Due to the nature of the use-case, sensitivity stands out as the most important metric and CNN shows scores higher than the threshold. High sensitivity score, combined with winner specificity score indicates a more robust model position for CNN among others.

With further tunning, data augmentation and more data pre-trained models could perform better than the CNN model.

