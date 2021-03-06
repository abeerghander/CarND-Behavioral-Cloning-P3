# Behavioral Cloning Project
**By Abeer Ghander**

[//]: # (Image References)

[image1]: ./cnn-architecture-624x890.png "Nvidia Model Visualization"
[image2]: ./Figure_A.png "Plot"


[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in the workspace:
* model.py
* drive.py
* video.py
* readme.md
* report.ipynb

The simulator can be downloaded from the classroom.
The training data were collected using the joystick (to have smoother driving). The training data had records for 3 laps in the center of the lane, 1 lap in the opposite direction, and 1.5 laps with correction from the side to the center (to help the network return back to the center whenever it deviates).

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is based on the [Nvidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). (model.py lines 64-85)
![alt text][image1]

The model summary is as follows:

|Layer (type)               |      Output Shape        |       Param #   |
|---------------------------|:------------------------:|----------------:|
|lambda_1 (Lambda)          |(None, 160, 320, 3)       |               0 |
|cropping2d_1 (Cropping2D)  |(None, 65, 320, 3)        |               0 |
|conv2d_1 (Conv2D)          |(None, 31, 158, 24)       |            1824 |
|conv2d_2 (Conv2D)          |(None, 14, 77, 36)        |           21636 |
|conv2d_3 (Conv2D)          |(None, 5, 37, 48)         |           43248 |
|conv2d_4 (Conv2D)          |(None, 3, 35, 64)         |           27712 |
|conv2d_5 (Conv2D)          |(None, 1, 33, 64)         |           36928 |
|dropout_1 (Dropout)        |(None, 1, 33, 64)         |               0 |
|flatten_1 (Flatten)        |(None, 2112)              |               0 |
|dense_1 (Dense)            |(None, 100)               |          211300 |
|dense_2 (Dense)            |(None, 50)                |            5050 |
|dense_3 (Dense)            |(None, 10)                |             510 |
|dense_4 (Dense)            |(None, 1)                 |              11 |
|**Total params: 348,219**                                                   |
|**Trainable params: 348,219**                                               |
|**Non-trainable params: 0**                                                 |


### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 78). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 150).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### 5. Data augmentation and generators

First the image is cropped to work only on the road-related part of the image. 

I used a generator to adapt the training data to work on 3 images: center image, left image, and right image (with corrected angles).

The validation data only used the center image.

I also used data augmentation to flip all images: all training data, and validation data. This was done to ensure that the model works on enough data, with different scenes. However, since the training time for each epoch was **HUGE**. I commented out the flipping code, and only sticked to the generators (with the 3 images for training data) and good quality training data.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to apply the well-known Nvidia network, to have robust training based on provided research.

Then, it was time to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I had to investigate what the problem is, until the bug in the drive.py was fixed where I needed to adapt input images to BGR instead of default RGB.

Also, it was essential to add some more training data where the car corrects its position from the side of the lane to the center of the lane. So, the training data was smooth, and with corrections of the car position from the side of the lane to the center of the lane.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself in case of error.

For the training data, I appended the left and right cameras with the corrected angle to enrich the dataset.

To augment the dataset, I also flipped images and angles thinking that this would increase the richiness of the dataset, where it has more scenes, and it would know how to react in more situations.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the neural network with 3 Epochs. The accuracy was very high after 2 rounds, so there was no need to increase the number of epochs more than 3. Here is the output of the epochs after running the model.

```
Epoch 1/3
34203/34203 [==============================] - 15629s 457ms/step - loss: 0.0030 - val_loss: 0.0078
Epoch 2/3
34203/34203 [==============================] - 15649s 458ms/step - loss: 0.0011 - val_loss: 0.0080
Epoch 3/3
34203/34203 [==============================] - 15626s 457ms/step - loss: 8.7662e-04 - val_loss: 0.0093
```
I used an adam optimizer so that manually training the learning rate wasn't necessary. Also, the `fit_generator()` function from Keras to run my training data and validation data.

```python
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, nb_epoch=nbEpoch,
                              validation_data=validation_generator, nb_val_samples=len(validation_samples))
```

## Details About Files In This Directory

### `model.py`
Model.py is where the model is implemented, augmentation and generation is done.

At the end of the run, the `model.h5` was generated using the following code snippet, and it is ready to be used for autonomous driving.

```sh
model.save(filepath)
```

Also, I generated a plot to view of the history of improvement in the accuracy along epochs:
![alt text][image2]

### `drive.py`

Usage of `drive.py` uses the trained model `model.h5`.  It can be used with drive.py using this command:

```sh
python drive.py model.h5 runA
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

In the beginning, the car did not work properly in the autonomous mode. But this was fixed by converting the images coming from the simulator to BGR instead of RGB. Since the model was trained on BGR images, the same format for images should be the input for the autonomous mode as well.

```python
image_array = image_array[:, :, ::-1].copy() #convert to BGR
```

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 runA
```

The fourth argument, `runA`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls runA

2019_02_13_09_52_18_005.jpg  
2019_02_13_09_52_18_196.jpg  
2019_02_13_09_52_18_351.jpg  
2019_02_13_09_52_18_512.jpg  
2019_02_13_09_52_18_703.jpg
2019_02_13_09_52_18_066.jpg  
2019_02_13_09_52_18_250.jpg  
2019_02_13_09_52_18_398.jpg  
2019_02_13_09_52_18_566.jpg  
2019_02_13_09_52_18_769.jpg
2019_02_13_09_52_18_125.jpg  
2019_02_13_09_52_18_301.jpg  
2019_02_13_09_52_18_462.jpg  
2019_02_13_09_52_18_636.jpg  
2019_02_13_09_52_18_828.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py runA
```

Creates a video based on images found in the `runA` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `runA.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py runA
```

[Video output](./runA.mp4)

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.


