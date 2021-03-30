# Personality-Classifier

This personality classifier project uses a combination of machine learning algorithms and a careful selection of facial features to precisely determine the personality of a person.

# Prerequisites

This project assumes an intermediate to advanced knowledge of Python and its data science libraries, including Numpy, Keras, Tensorflow, and Pandas. Additionally knowledge of CNNs and how object detection algorithms will be useful is helpful.

## Dependencies

- Python: >3.7
- Tensorflow: >2.4
- Pandas: >1.2
- OpenCV: >4.0
- dlib: 19.21.1
- Numpy: >1.20
- imutils: >0.5

# Dataset

The dataset that was used was the CelebA dataset, which provides a dataset with over 200,000 images of human faces with 41 different facial characteristics like arched eyebrows, nose size, etc. Some of these characteristics do overlap with the characteristics found in the dlib library; however, we can simply ignore them for our purposes.

Here is the link: [https://www.kaggle.com/jessicali9530/celeba-dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)

Additionally, to allow for greater accuracy and more data points, unlabeled data was also collected to form a semi-supervised image classifier. To get these unlabeled images, images from Instagram and Twitter were scraped from using the facial characteristics identified throughout the face. These images were collected using a particular hashtag for the facial characteristic (e.g. arched eyebrows would be #archedeyebrows).

## Filtering

Because scraping can often produce unreliable results, there has to be a way to distinguish faces that are reliable for analysis from those that will not work. To do this, the dlib library feature mentioned below can be used. If a bounding box does not exist, or the 68 points cannot be found, that means that there is not a face that can be used, and the image is discarded.

This is a great way to filter through images, as any image that isn't a face is immediately discarded before it is put through the trained network.

# Step #1: Mapping the Face

Before figuring out an individual's personality, the first step is to map different elements of the face to determine facial features. Once determining the facial features, we can then make assumptions about the personality.

To do this, the dlib library is used, which creates a framework for facial landmark detection like so ([https://miro.medium.com/max/700/1*mmrx-sennJ2N3r59NUxouQ.png](https://miro.medium.com/max/700/1*mmrx-sennJ2N3r59NUxouQ.png)) This consists of 68 points that are placed on specific regions of the face (left eye area, right eye area, bridge of nose, etc.) By using this, we can get general proportions of the face and use that for facial feature analysis.

For instance, wide set eyes require that the distance between the bridge of the nose and the right side of the left eye is greater than your eye length. By utilizing this information and the 68 points, we should be able to accurately identify facial features.

## How Does It Work?

The dlib library uses a bounding box feature to first recognize distinct faces in the image using Haar cascades. Next, within that box, it runs a HOG algorithm or other similar object-detection algorithms to identify coordinates on a person's face. 

## Importance

We can know identify a variety of different parts of the face in our image, which will allow us to make more accurate predictions about facial features. 

This will allow us to identify even more facial features. For example, we can create a bounding box around the eye and determine eye color now. This would've been difficult to do if we didn't know the color of the eye.

## Difficult Facial Features

For difficult facial features to determine from proportions, we can use CNNs or convolutional neural networks. To train this massive neural network, we would have to use a lot of data; this data can be provided from Instagram scraping or from the CelebA dataset. 

Some examples of facial features that can be found are the following: arched eyebrows, nose size, and bags under eyes.

To add accuracy, a separate convolutional neural network will be used for each facial feature, and the final output will be a binary classification of whether or not the facial feature is there or not. 

Because some images in the dataset are more reliable than others (e.g. the scraped images are less reliable than those from the CelebA dataset), we add different weights to each of these images when training the model with these batches.

# Step #2: Classifying the Individual's Personality
 
After finding these facial characteristics using ML algorithms and the *dlib* library, the personality can easily be classified by using a lookup table containing the facial characteristic and its corresponding strength pertaining to the personality trait.

## Lookup Table

The lookup table will act as a Pandas DataFrame with the individual facial characteristic on the first column and then the subsequent strength value depending on the personality trait on a -1 to 1 scale.

Two things to take note of:

- The closer the value is to one, the greater the correlation is between the facial characteristic and the personality trait.
- If the value is negative, then it exhibits the opposite of the personality trait.

For example, having a light blue eye color gives a personality trait strength of -1 for having a short temper, that means that the person is extremely patient. 

Taking the average of these different facial characteristics, a definite personality can be identified.
