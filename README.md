# Nanotechnology-and-Machine-Learning


**Product Mission:**

Our mission is to conduct research to discover ways in which machine learning can be used in conjunction with nanotechnology to discover and improve cancer prediction and diagnosis methodologies. Through the exploration of this area, our goal is to contribute to the field of cancer research by understanding how nanotechnology is currently used in the field and how machine learning can be used to improve the current capabilities of nanotechnology.

<br />

**Minimum Viable Proudct:**

As a user, I want to:

-	Be able to use the designed ML model in skin cancer detection.

-	Explore how nanotechnology is used in cancer prediction and diagnosis. 

-	Design theoretical experiments using different nanotechnologies and integrate them with the designed ML model.

<br />

Our MVP is to design a basic ML model for skin cancer detection, followed by researching different ways in which we can use our ML model with nanotechnology. This involves the exploration of current methods which utilize nanotechnology in cancer prediction and diagnosis, and selecting which pieces of nanotechnology we would like to use in our own study. Following this, we will attempt to design theoretical experiments using our ML model and specific pieces of nanotechnology. 

<br />

**Technologies:**

- Machine learning models (supervised and unsupervised)
- Nanotechnology (emphasis on detection and imaging) - Carbon Nanotubes
- Datasets - Kaggle
- Project MONAI (open-source framework)



# Sprint 2 Skin Cancer Diagnosis by using Machine Learning

- Pre-processing
- Segmentation
- Feature extraction
- Classification


![skin cancer detect](https://user-images.githubusercontent.com/7721258/139742308-af1d54b1-2182-4dcb-98d0-73e465ec2a34.jpg)



# Sprint 3:

**Attempts to generate synthetic/artificial data for our ML model.**

- Find relevant literature (e
- Interpret graphs, charts, etc.
- Parse information to create training datasets


**Table from "Carbon nanotubes in cancer diagnosis and therapy**:

![Parsing information 1](https://user-images.githubusercontent.com/56008239/141825905-7e2ad095-43c0-4168-8c81-cb71275127eb.jpg)


**Creating our initial ML model for detection and training**

# Data Split
Because the data provided is a whole dataset, we should split it into 2 sets, one for training and another for testing. In this step we will split the dataset into training and testing set of 80:20 ratio

# Step of Model Building
**CNN (Convolutional Neural Networks)**


I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.
![image](https://user-images.githubusercontent.com/7721258/142282969-1cd0af7b-cec8-4b7d-bc51-f289c88a83cc.png)


Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.
![image](https://user-images.githubusercontent.com/7721258/142285866-2af8ebf0-68b4-4919-9924-eb82815a0db6.png)


'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

**Image of Model** 

![8c5af6a1ef92b082f9f6a69feef7fc8](https://user-images.githubusercontent.com/87682737/141837496-730ad9fc-3ce9-4ae0-86dc-de1820a7a225.png)

# Setting Optimizer

The most important function(I think) is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss. I choosed Adam optimizer because it combines the advantages of two other extensions of stochastic gradient descent. Specifically:

1. Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).

2. Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

Adam realizes the benefits of both AdaGrad and RMSProp. So, it's a popular algorithm in the field of deep learning because it achieves good results fast.

**Image of Optimizer** 

![0f1a826be50ad6e25add45d9c6fd45d](https://user-images.githubusercontent.com/87682737/141838254-14938ea1-d9e4-4919-b4f8-5625021818fd.png)

# Model Evalution

In this step we will check the testing accuracy and validation accuracy of our model,plot confusion matrix and also check the missclassified images count of each type
![17b18420de1fe290a96bc60e1057613](https://user-images.githubusercontent.com/87682737/142267946-4f553870-3c63-4e53-9863-69d0a8af8c99.png)


![image](https://user-images.githubusercontent.com/7721258/142285334-d9cb1c86-2531-40b9-8c2b-ba2b5112d182.png)

**References**

X. Li, J. Wu, E. Z. Chen, and H. Jiang, “From deep learning towards finding skin lesion biomarkers,” 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2019. 


# Sprint 4

**Goals**
- Create  signals datasets 
- Train the ML model to recognize generated signals
 
 
**Initial approach** 
- Finding data where carbon nanotubes are used as sensors (focusing on biomarkers)
- Skin cancer detection: Apart from physical appearance, what are the other indicators?
- What literature used carbon nanotubes to detect these specific biomarkers? 
- Convert to data 

![Potential MM biomarkers](https://user-images.githubusercontent.com/56008239/144278346-8d5e3b65-4c16-41b7-b7ff-e3d4a21ad35a.png)


**Creating Datasets: Limitations**

- The primary limitation with our initial approach is finding relevant data for use. Regarding skin cancer, carbon nanotubes were used specifically as a drug-delivery approach rather than detecting skin cancer. 
