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



**Sprint 2 Skin Cancer Diagnosis by using Machine Learning**

- Pre-processing
- Segmentation
- Feature extraction
- Classification


![skin cancer detect](https://user-images.githubusercontent.com/7721258/139742308-af1d54b1-2182-4dcb-98d0-73e465ec2a34.jpg)



**Sprint 3: Next Steps**

![image](https://user-images.githubusercontent.com/56008239/139744616-f97e75a8-e806-4282-9fe4-42f2b2716c50.png)


**Table from "Carbon nanotubes in cancer diagnosis and therapy**:

![Parsing information 1](https://user-images.githubusercontent.com/56008239/141825905-7e2ad095-43c0-4168-8c81-cb71275127eb.jpg)

# Step of Model Building
**CNN**
I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.
