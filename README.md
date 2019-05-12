# Traffic Sign Classifier

### Project #3 for Udacity Self-Driving Car Nanodegree

#### By Adam Gyarmati

---

The goal of this project was to build a neural network to recognize and classify images of traffic signs.
The project consisted of the following steps:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test the model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_train.png "Training data"
[image2]: ./examples/visualization_valid.png "Validation dataa"
[image3]: ./examples/visualization_test.png "Test data"
[image4]: ./examples/1_slippery.PNG "Traffic Sign 1"
[image5]: ./examples/2_stop_snowy.PNG "Traffic Sign 2"
[image6]: ./examples/3_priority_road.PNG "Traffic Sign 3"
[image7]: ./examples/4_stop.PNG "Traffic Sign 4"
[image8]: ./examples/5_speed_limit.PNG "Traffic Sign 5"
[image9]: ./examples/softmax_probabilities.png "Softmax probabilites"

---
### Writeup, Ipython notebook and HTML output

Beside this writeup, the directory contains the Ipython notebook and HTML output of the code.

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used python methods to calculate the following summary statistics of the traffic signs data set:

* Size of the training set: 34799
* Size of the validation set: 4410
* Size of test set: 12630
* Shape of a traffic sign image: 32, 32, 3
* Number of unique classes/labels in the data set: 43

#### 2. Include an exploratory visualization of the dataset.

Visualization of the dataset consists of three histograms showing the distribution of images between the different classes:

Training data:
![alt text][image1]

Validation data:
![alt text][image2]

Test data:
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Pre-processing the Data Set (normalization and grayscale)

I chose to grayscale the images after training my network on colored and grayscale and finding no significant difference. Therefore, to speed up the training I decided to use grayscale images

Images are grayscaled simply by averaging the RGB values. Next the images are normalized using (pixel - 128)/ 128 so that the data has a mean close to zero and equal variance.

#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, output: 28x28x6 	|
| RELU					|												|
| Max pooling   		| 2x2 stride,  output: 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, output: 10x10x16 	|
| RELU					|												|
| Max pooling   		| 2x2 stride,  output: 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, output: 1x1x400 	|
| RELU					|												|
| Flatten				| Takes conv2 and conv3, output: 400 + 400 		|
| Concatenate			| Concatenate flattened layers, output: 2000	|
| Dropout				| Throws half of the weights out randomly		|
| Fully connected		| Input: 2000, output: 43						|
 

#### 3. Training the neural network

To train the model, I used the Adam optimizer recommended for this project with a learning rate of 0.001, a batch size of 128, and trained the neural network for 30 epochs. I didn't experiment much with the hyperparameters, because this takes a lot of GPU time and I was able to acheive the required validation accuracy by modifying the model architecture. Surely the training parameters could be improved to get a higher accuracy, hopefully I'll have the time to come back to this project after the end of the course... :)

#### 4. Results & discussion

My final model results were:

* training set accuracy of ~1.00
* validation set accuracy of 0.961 
* test set accuracy of 0.937

I ended up with the current model architecture after the following steps:

1) I first implemented the LeNet-5 architecture shown in the course
* With this initial architecture, I acheived 0.93 validation accuracy with 80 epochs
* By adding a dropout layer before the fully connected layer, I acheived 0.94 validation accuracy with 80 epochs
* One problem with the inital architecture is that training took very long because of the high number of epochs
2) Therefore, I changed the model according to the [ConvNet architecture](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
* This approach takes the output of the first convolutional layer and feeds it directly to the classifier
* For me, taking the output of the second convolutional layer worked better therefore I changed it to feed *conv2* to the classifier
* With this approach, I reached 0.961 validation accuracy


### Testing the model on new images

#### 1. Testing the model on five German traffic signs found on the web

these are the five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I tried to find some images which are difficult to classify to see whether my model can deal with them. I ended up with:
* 2 snowy traffic signs
* 2 tilted traffic signs
* 1 fairly obvious and easily distinguishable sign

#### 2. Predictions

Results of the prediction:

![alt text][image9]

The classifier did better than I expected, and could classify most of the traffic signs (4 out of 5 = 80%) correctly. The most difficult was the snowy *slippery road* sign, which is understandable because half of the sign is covered with snow :) Still the model was sometimes able to classify it correctly and the correct sign was always among the top softmax probabilities.

#### 3. Softmax probabilities for the downloaded images

For the first and most difficult image, the softmax probability is 0%, which means unfortunately the model wasn't even close... However it is interesting to see that the other guesses are mainly triangular shaped, which at least tells us the type of traffic sign (warning/hazard ahead).

For the rest of the images, the softmax probabilites are all 100% and the classification is correct.
