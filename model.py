import csv
import cv2
from os import path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadData(basePath):
    """ Load the images and steering angles from the basePath
        assuming the headline of the CSV file is removed (starting the image name)
    :param basepath: base path which contains the csv file and IMG folder
    :return: lines array of the csv file containing the data
    """
    lines = []
    first_line = True
    with open(path.join(basePath,'driving_log.csv')) as f:
        content = csv.reader(f)
        for line in content:
            if first_line is False:
                lines.append(line)
            else:
                first_line = False

    return lines

'''
def balance_data(samples, visulization_flag ,N=60, K=1,  bins=100):
    """ Crop the top part of the steering angle histogram, by removing some images belong to those steering angels
    :param images: images arrays
    :param angles: angles arrays
    :param n:  The values of the histogram bins
    :param bins: The edges of the bins. Length nbins + 1
    :param K: maximum number of max bins to be cropped
    :param N: the max number of the images which will be used for the bin
    :return: images, angle
    """

    angles = []
    for line in samples:
        angles.append(float(line[3]))

    n, bins, patches = plt.hist(angles, bins=bins, color= 'orange', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)

    idx = n.argsort()[-K:][::-1]    # find the largest K bins
    del_ind = []                    # collect the index which will be removed from the data
    for i in range(K):
        if n[idx[i]] > N:
            ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            del_ind.extend(ind[:len(ind)-N])

    balanced_samples = [v for i, v in enumerate(samples) if i not in del_ind]
    balanced_angles = np.delete(angles,del_ind)

    plt.subplot(1,2,2)
    plt.hist(balanced_angles, bins=bins, color= 'orange', linewidth=0.1)
    plt.title('modified histogram', fontsize=20)
    plt.xlabel('steering angle', fontsize=20)
    plt.ylabel('counts', fontsize=20)

    if visulization_flag:
        plt.figure
        plt.subplot(1,2,1)
        n, bins, patches = plt.hist(angles, bins=bins, color='orange', linewidth=0.1)
        plt.title('origin histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

        plt.figure
        aa = np.append(balanced_angles, -balanced_angles)
        bb = np.append(aa, aa)
        plt.hist(bb, bins=bins, color='orange', linewidth=0.1)
        plt.title('final histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

    return balanced_samples
'''

'''
def brightness_change(image):
    """  change the brightness of the input image
    :param image: input image
    :return: new image
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    #print(image1.shape)
    return image1
'''

def data_augmentation(images, angles):
    """ flip every image and change the brightness of the image, then appended to the lists
    :param images: origin image
    :param angles: origin angles
    :return: added augmented images and their angles
    """
    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):

        augmented_images.append(image)
        augmented_angles.append(angle)

        # flip
        flipped_image = cv2.flip(image,1)
        flipped_angle = -1.0 * angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)

        # brightness changes
        #image_b1 = brightness_change(image)
        #image_b2 = brightness_change(flipped_image)

        # append images
        #augmented_images.append(image_b1)
        #augmented_angles.append(angle)
        #augmented_images.append(image_b2)
        #augmented_angles.append(flipped_angle)

    return augmented_images, augmented_angles


def network_model():
    """
    :return: designed network model
    """

    # Define the model, adapted version of Nvidia model with an additional dropout layer
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    '''    
    #model = Sequential()
    #model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    ch, row, col = 3, 160, 320  # Trimmed image format

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256,3,3, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(20))
    model.add(Dense(1))
    '''
    '''
    model = Sequential()
    # Normalizing the input image data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(32,80,3)))
    # First Convolution2D layer
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    # Second Convolution2D layer
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D())
    # Flattening the output of 2nd Convolution2D layer
    model.add(Flatten())
    # First Dense layer
    model.add(Dense(32))
    model.add(Dropout(0.20))
    # Second Dense Layer
    model.add(Dense(16))
    # Third and Final Dense Layer
    model.add(Dense(1))
    '''
    return model

'''
# Method to pre-process the input image
def pre_process_image(image):
    # Since cv2 reads the image in BGR format and the simulator will send the image in RGB format
    # Hence changing the image color space from BGR to RGB
    #colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Cropping the image
    #cropped_image = colored_image[60:140, :]
    # Downscaling the cropped image
    #resized_image = cv2.resize(cropped_image, None, fx=0.25, fy=0.4, interpolation=cv2.INTER_CUBIC)
    #return resized_image
    return image
    #return colored_image
'''

def generator(basePath, samples, train_flag, batch_size=32):
    """
    """
    num_samples = len(samples)
    correction = 0.2  # correction angle used for the left and right images

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for line in batch_samples:
                angle = float(line[3])
                c_imagePath = line[0].replace(" ", "")
                c_imagePath = basePath + c_imagePath
                #print(c_imagePath)
                c_image = cv2.imread(c_imagePath) #BGR format
                #processed_c = pre_process_image(c_image)
                #images.append(processed_c)
                images.append(c_image)
                angles.append(angle)

                if train_flag:  # only add left and right images for training data (not for validation)
                    l_imagePath = line[1].replace(" ", "")
                    l_imagePath = basePath + l_imagePath
                    r_imagePath = line[2].replace(" ", "")
                    r_imagePath = basePath + r_imagePath
                    l_image = cv2.imread(l_imagePath)
                    r_image = cv2.imread(r_imagePath)

                    #processed_l = pre_process_image(l_image)
                    #processed_r = pre_process_image(r_image)
                    
                    #images.append(processed_l)
                    images.append(l_image)
                    angles.append(angle + correction)
                    #images.append(processed_r)
                    images.append(r_image)
                    angles.append(angle - correction)

            # flip image and change the brightness, for each input image, returns other 3 augmented images
            #augmented_images, augmented_angles = data_augmentation(images, angles)

            #X_train = np.array(augmented_images)
            #y_train = np.array(augmented_angles)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# load the csv file
basePath = '/home/carnd/CarND-Behavioral-Cloning-P3/data/'
print('loading the data...')
#print('basepath',basePath)
samples = loadData(basePath)
print('Finished loading the csv data... ', len(samples), " lines are loaded.")

# balancethe data with smooth histogram of steering angles
#samples = balance_data(samples, visulization_flag=False)
#print('balancing the data finished...')

# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(basePath, train_samples, train_flag=True, batch_size=32)
validation_generator = generator(basePath, validation_samples, train_flag=False, batch_size=32)

# define the network model
model = network_model()
model.summary()

nbEpoch = 3 # first time tried with 4 epochs, it took forever, and it was already having minimal loss from the second epoch
model.compile(loss='mse', optimizer='adam')

print("len(train_samples): ", len(train_samples))
print("len(validation_samples): ", len(validation_samples))

#model.compile(loss='mse', optimizer='adam')
#print("Training...")
#history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=5, verbose=1)

#history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*12, nb_epoch=nbEpoch, 
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, nb_epoch=nbEpoch,                              validation_data=validation_generator, nb_val_samples=len(validation_samples)*2)

model.save('model.h5')

### print the keys contained in the history object
print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()