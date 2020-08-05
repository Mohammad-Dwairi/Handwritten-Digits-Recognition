import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import requests
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

random.seed(0)


# Create and display a gird containing samples from each class (Number) that were imported from MNIST
def create_samples_grid():
    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    fig.tight_layout()

    # Fill the cells of the grid with the appropriate image, We want to show 10 samples from each class (10 Classes)
    for i in range(10):
        for j in range(10):
            # Select all images where the label is = j
            selected = training_data[training_labels == j]

            # Show random images from the selected images on the grid
            axes[j][i].imshow(selected[random.randint(0, len(selected) - 1), :, :], cmap=plt.get_cmap("gray"))

            # Remove axis around images
            axes[j][i].axis("off")

            # Display the labels for the classes at column 4 (The center of the grid)
            if i == 4:
                axes[j][i].set_title(str(j))

                # Store how many images (Samples) we have for each class
                num_of_samples.append(len(selected))


def visualize_samples_info():
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 10), num_of_samples)
    plt.title("Distribution of the training data sets")
    plt.xlabel("Class Number")
    plt.ylabel("Number of Images")


def create_model():
    # Linear stack of layers.
    s_model = Sequential()

    # Just a regular densely-connected neural network layer.
    # 10 is the number of possible output after processing the image (The prediction 0 -> 9)
    # input_dim=784 is the number of input nodes in the input layer (784 is the number of pixels in 28x28 image)
    # therefore, we flattened our 28x28 pixels array into 784 one dimensional array
    # The activation function is responsible for transforming the summed weighted input
    # from the node into the activation of the node or output for that input.
    s_model.add(Dense(10, input_dim=784, activation='relu'))

    # Second Layer of the neural network (Adding more layers can lead to overfit the model)
    s_model.add(Dense(20, activation='relu'))

    # Output layer
    s_model.add(Dense(10, activation='softmax'))

    # Configures the model for training
    # Adam - A Method for Stochastic Optimization
    # Cross Entropy is a loss function for Classification Problems that minimizes the distance between
    # two probability distributions - predicted and actual
    s_model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return s_model


#######################################################################################################################

#  Returns 60K training sets and 10k test sets with their labels
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()

# The number of samples returned from MINST for each class (EX: 6000 image sample from number (Class) 0 )
num_of_samples = []

# Insures that we have a clean data from MNIST
assert (training_data.shape[0] == training_labels.shape[0]), "The number of images is not equal to the number of labels"
assert (test_data.shape[0] == test_labels.shape[0]), "The number of images is not equal to the number of labels"
assert (training_data.shape[1:] == (28, 28)), "The dimensions of the images is not 28x28"
assert (test_data.shape[1:] == (28, 28)), "The dimensions of the images is not 28x28"

create_samples_grid()
visualize_samples_info()

# One hot encoding is a process by which categorical variables are converted into a form
# that could be provided to ML algorithms to do a better job in prediction.
# 0 indicates non existent while 1 indicates existent.
training_labels = to_categorical(training_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Normalize data to range between 0 and 1 (each pixel intensity ranges between 0 and 255)
training_data = training_data / 255
test_data = test_data / 255

# Reshaping the data sets into 60K images each has 784 (28x28) pixel as one dimensional array (flattened)
# Preparing it to be entered into input layer (first layer) of the neural network
training_data = training_data.reshape(training_data.shape[0], 784)
test_data = test_data.reshape(test_data.shape[0], 784)

########################################################################################################################

model = create_model()

# Trains the model for a fixed number of epochs (iterations on a datasets)
# Number of samples per gradient update
# validation_split: Fraction of the training data to be used as validation data
# The model will set apart this fraction of the training data, will not train on it, and will evaluate
# the loss and any model metrics on this data at the end of each epoch.
history = model.fit(training_data, training_labels, validation_split=0.1, epochs=10, batch_size=200)


plt.figure(figsize=(12, 4))
# Returns the loss history when training the model on the training datasets
plt.plot(history.history['loss'])
# Returns the loss history when training the model on the validation datasets, which the model hasn't trained on before
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epoch')

# Returns the loss value & metrics values for the model in test mode
score = model.evaluate(test_data, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Number (3) image
url = "https://www.neuralnine.com/wp-content/uploads/2019/09/3.png"

# HTTP Get request to get the image from the url
response = requests.get(url, stream=True)

# Open the image and store it as array of pixels
img = Image.open(response.raw)
img = np.asarray(img)

# Resize the image to (28 pixels X 28 pixels) to match the samples that out model trained on
img = cv.resize(img, (28, 28))

# Convert the image to Grayscale to reduce the color channels and computational time
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Convert (White Background, Black Digit) -> (Black Background, White Digit)
img = cv.bitwise_not(img)
plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))

img = img / 255
img = img.reshape(1, 784)

prediction = model.predict_classes(img)
print("predicted digit:", str(prediction[0]))

plt.show()
