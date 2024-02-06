
# Faultfindy: Faulty Tyre Classification

### Project Introduction
Overview In the era of rapid technological advancements, the application of machine learning in various industries has become increasingly pivotal. This project focuses on leveraging the power of deep learning for image classification, specifically in the context of the automotive industry. We aim to develop a model capable of distinguishing between defective and good condition tyres based on digital images. This task is not only technologically intriguing but also carries significant implications for vehicle safety and quality control in tyre manufacturing.
### Dataset
This dataset digital images of tyres, divided into two categories: 1,028 defective and 828 in good condition. Each high-resolution image is meticulously labelled to indicate the tyre's state. This comprehensive collection is ideal for machine learning and computer vision applications, specifically in image classification and object detection. It provides a balanced mix of conditions, offering a robust resource for algorithm training and testing in identifying tyre conditions from digital images. 
(Data Set link: https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated Project guide data set/Faultfindy.zip)
![App Screenshot](https://i.ibb.co/H2p77qB/Tyre-Condition-Count-Plot.png) ![App Screenshot](https://i.ibb.co/sjq0Pbs/observed-category-proportion.png)

### Objectives
Develop an Image Classification Model: Utilize deep and Machine learning algorithms to accurately classify images of tyres into two categories: defective and good condition. Compare Different Architectures: Evaluate the performance of different models like **KNN**,**Random Forest**,**VGG16** and **ResNet50** to understand the trade-offs between model complexity and accuracy. Optimize Model Performance: Employ advanced techniques like dynamic learning rate adjustment and early stopping to fine-tune the models for optimal performance. Significance The successful implementation of this project can lead to several key advancements:

Enhancing Safety Standards: By accurately identifying defective tyres, the model can contribute to improved safety measures in the automotive industry. Quality Control in Manufacturing: The automated detection of tyre defects can streamline quality control processes in tyre production. Technological Benchmarking: This project serves as a benchmark for applying deep learning techniques in practical, real-world scenarios, particularly in industries where safety is paramount. With these objectives and significance in mind, let's delve into the specifics of the project, starting with data preprocessing and manipulation.
Let's have a look on few sample images from both the categories:
![App Screenshot](https://i.ibb.co/RB2V2VQ/explore-data-Set.png)

### Lets plot the Training and Validation Loss Curves for deep learning models "VGG16" and "ResNet-50" 
![App Screenshot](https://i.ibb.co/LC3xR2L/Training-Validation-Accuracy-for-VGG16.png)

![App Screenshot](https://i.ibb.co/0Vv9y3f/Training-Validation-Accuracy-for-Res-Net-50.png)

### Now analyze the confusion matrix
## *VGG16
![App Screenshot](https://i.ibb.co/Ydj0wT8/vgg16-Confusion-Matrix.png))  
## *ResNet-50
![App Screenshot](https://i.ibb.co/F0gxmMW/resnet-50-Confusion-Matrix.png)
## *KNN
![App Screenshot](https://i.ibb.co/D77t5y2/KNN-Confusion-Matrix.png) 
## *Random Forest 
![App Screenshot](https://i.ibb.co/HBdjXyW/Random-Forest-Confusion-Matrix.png)

### Compare the accuracy of ***KNN***,***Random Forest***,***VGG16*** and ***ResNet50*** to find the best model
![App Screenshot](https://i.ibb.co/YZkyK35/best-model-vgg16.png)

# It clearly shows that VGG16 is perforing best with an accuracy of 93% 






