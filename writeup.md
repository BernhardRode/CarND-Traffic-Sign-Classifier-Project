# **Traffic Sign Recognition** 

## Writeup

---

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/colored_signs.png "Grayscaling"
[image3]: ./examples/normalized.png "Normalized"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/BernhardRode/CarND-Traffic-Sign-Classifier-Project)

See the final Version in Traffic_Sign_Classifier.html or run the code in Traffic_Sign_Classifier.ipynb.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the no library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
34799
* The size of the validation set is ?
TODO
* The size of test set is ?
12630
* The shape of a traffic sign image is ?
32x32x3 - 32x32 pixels in RGB color
* The number of unique classes/labels in the data set is ?
43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

TODO

![Visualization][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first the image dataset looks like this:

![Traffic signs][image2]

I tried several approaches from the previous project (Lane detection). But none 
of them helped alot. At the end I just used a normalize function which normalizes 
the input images to values between 0 and 1. This led to the best results so far.

After getting a typ from a friend, I extended the simple normalization function a
little bit and return values between -0.5 and 0.5. This reduces 

![Normalized traffic signs][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Following from the advices from the lessoon, I used the adapted LeNet architecture
from the last lab.

| Layer         		|     Input/Output	        					| Parameter | 
|:---------------------:|:-------------------------------------------------:|:-------------------------------:|
| Input         		| 32x32x3 RGB image   							| 
| Convolutional | Input 32x32x3 Output = 28x28x6 | strides=[1, 1, 1, 1], padding='VALID' |
| RELU | | |
| Max Pool | Input = 28x28x6. Output = 14x14x6 | ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID' |
| Convolutional | Input = 10x10x16 Output = 14x14x6 | strides=[1, 1, 1, 1], padding='VALID' |
| RELU | | |
| Max Pool | Input = 10x10x16 Output = 5x5x16 | ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID' |
| Flatten | Input = 5x5x16. Output = 400 | |
| Connected | Input = 400. Output = 120 | |
| RELU | | |
| Dropout | | DROP_RATE |
| Connected | Input = 120 Output = 84 | |
| RELU | | |
| Dropout | | DROP_RATE |
| Connected | Input = 84. Output = 10 | |
| Output | | |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used these base parameters to train the model:

EPOCHS = 7
BATCH_SIZE = 128
KEEP_PROB = 0.80
RATE = 0.001

Following the lab, I used the AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My Accuracies for the Epochs:

EPOCH 1 ...
Training Accuracy = 0.648
Validation Accuracy = 0.618

EPOCH 2 ...
Training Accuracy = 0.878
Validation Accuracy = 0.791

EPOCH 3 ...
Training Accuracy = 0.942
Validation Accuracy = 0.872

EPOCH 4 ...
Training Accuracy = 0.962
Validation Accuracy = 0.885

EPOCH 5 ...
Training Accuracy = 0.974
Validation Accuracy = 0.917

EPOCH 6 ...
Training Accuracy = 0.977
Validation Accuracy = 0.910

EPOCH 7 ...
Training Accuracy = 0.985
Validation Accuracy = 0.920

My final model results were:
* training set accuracy of ?
0.985
* validation set accuracy of ? 
0.920
* test set accuracy of ?
0.904

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

At first, I tried the basic LeNet from the lab, which gave around 84 % of training accuracy.

* What were some problems with the initial architecture?

I need around 10-13 Epochs to get to a point, where I'm above 85 % training accuracy, which takes a lot of time.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

A long time, I forgot to normalize the images in my validation set. So I just got to around 60 % accuracy there. =)

After finding this bug, I was really pleased with the result so far.

* Which parameters were tuned? How were they adjusted and why?

Drop out rate and learning rate made just differences of 2-3 % in the end. 

I played around with different Batch Sizes, Epochs to speed up the training time.

The biggest impact was adapting the normalizing function to return -0.5 to 0.5
without this, i was just able to get to 85 % no matter what I tried. This is 
why I ended up using just this approach. For me it was the best trade off 
between speed and result.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?

LeNet 

* Why did you believe it would be relevant to the traffic sign application?

To be true... I trusted in you ;)

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

With 98.5 % accuracy on the training set, this is imho a really good end result.
The gap between training and validation (92.0%) and test (90.4%) may be a clue, 
that the model is overfitting on the training data a little bit too much.

Up to now I'm still missing the gut feeling to say if this difference is good or bad.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

TODO

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

TODO

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

TODO

