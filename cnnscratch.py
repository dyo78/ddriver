import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.layers import Layer
from scipy.signal import correlate2d

# Define paths to train and test folders
train_folder = r'C:\Users\diyam\Documents\datase\Train'
test_folder = r'C:\Users\diyam\Documents\mrlEyes_2018_01\Prepared_Data\Test'

classes = ['open', 'closed']

X_train = []
y_train = []
X_test = []
y_test = []

img_size = (92, 112)

for i, class_name in enumerate(classes):
    class_path = os.path.join(train_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        X_train.append(img)
        y_train.append(i)

    class_path = os.path.join(test_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        X_test.append(img)
        y_test.append(i)

X_train = np.array(X_train, dtype=np.uint8)
y_train = np.array(y_train)
X_test = np.array(X_test, dtype=np.uint8)
y_test = np.array(y_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

learning_rate = 0.001  # Define the learning rate
# Define custom Conv2D Layer
class Conv2D(Layer):
    
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters
        
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)
    def forward(self, input_data):
        self.input_data = input_data
        # Initialized the input value
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            output[i] = correlate2d(self.input_data, self.filters[i], mode="valid")
        #Applying Relu Activtion function
        output = np.maximum(output, 0)
        return output 
    def backward(self, dL_dout, lr):
        # Create a random dL_dout array to accommodate output gradients
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
                # Calculating the gradient of loss with respect to kernels
                dL_dfilters[i] = correlate2d(self.input_data, dL_dout[i],mode="valid")

                # Calculating the gradient of loss with respect to inputs
                dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")

        # Updating the parameters with learning rate
        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        # returning the gradient of inputs
        return dL_dinput
    

class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):

        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):

                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    # Creating a patch from the input data
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    #Finding the maximum value from each patch/window
                    self.output[c, i, j] = np.max(patch)

        return self.output
    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dL_dinput[c,start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput

class Dense:
 
    def __init__(self, input_size, output_size):
        self.input_size = input_size # Size of the inputs coming
        self.output_size = output_size # Size of the output producing
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)
    def softmax(self, z):
        # Shift the input values to avoid numerical instability
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        # Compute the softmax probabilities
        probabilities = exp_values / sum_exp_values

        return probabilities
    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)
    def forward(self, input_data):
        self.input_data = input_data
        # Flattening the inputs from the previous layer into a vector
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        # Applying Softmax
        self.output = self.softmax(self.z)
        return self.output
    def backward(self, dL_dout, lr):
        # Calculate the gradient of the loss with respect to the pre-activation (z)
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)
        # Calculate the gradient of the loss with respect to the weights (dw)
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))

        # Calculate the gradient of the loss with respect to the biases (db)
        dL_db = dL_dy

        # Calculate the gradient of the loss with respect to the input data (dL_dinput)
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        # Update the weights and biases based on the learning rate and gradients
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        # Return the gradient of the loss with respect to the input data
        return dL_dinput


class Dropout:

  def __init__(self, rate):
    self.rate = rate

  def forward(self, X, training=None):
    if training is None:
      training = True  # Default to training mode if not specified
    if not training:
      return X  # No dropout during testing or evaluation

    # Create a mask with probability (1 - rate) of keeping neurons
    self.mask = np.random.rand(X.shape) > self.rate
    # Scale the output by the inverse of the dropout rate to maintain mean activation
    return X * self.mask / (1 - self.rate)

  def backward(self, d_out):
    # Implement backpropagation for gradient scaling during training (not shown here)
    return d_out * self.mask / (1 - self.rate)  # Scale gradients using mask
    
class Flatten:
  def __init__(self):
    pass

  def forward(self, X):
    self.X = X
    return X.reshape(-1, np.prod(X.shape[1:]))  # Efficient reshaping



def create_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(112, 92, 1)),
        MaxPooling2DLayer(pool_size=(2, 2), strides=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3)),
        MaxPooling2DLayer(pool_size=(2, 2), strides=(2, 2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(128, activation='relu', in_features=np.prod(Flatten().forward(X_train).shape[1:])),  # Add in_features
        Dropout(0.5),
        Dense(2, activation='softmax', in_features=128)  # Specify out_features
    ])
    return model

model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=20, 
      # You can increase the number of epochs for better training with a larger dataset
    validation_data=(X_test, y_test))

# Save the model
model.save('models/newfinaldiya1.keras', overwrite=True)
model.save('models/newfinaldiya1.h5')