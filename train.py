import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.layers import Layer

# Define paths to train and test folders
train_folder = r'C:\Users\diyam\Documents\mrlEyes_2018_01\Prepared_Data\Train'
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


# Define custom Conv2D Layer
class Conv2D(Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        output = K.conv2d(inputs, self.kernel, strides=(1, 1), padding='valid')
        output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = K.relu(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - self.kernel_size[0] + 1, input_shape[2] - self.kernel_size[1] + 1, self.filters)


class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size, strides, **kwargs):
        super(MaxPooling2DLayer, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs):
        return K.pool2d(inputs, pool_size=self.pool_size, strides=self.strides, padding='valid', pool_mode='max')


class DropoutLayer(Layer):
    def __init__(self, rate, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return K.dropout(inputs, level=self.rate)


def create_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(112, 92, 1)),
        MaxPooling2DLayer(pool_size=(2, 2), strides=(2, 2)),
        DropoutLayer(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3)),
        MaxPooling2DLayer(pool_size=(2, 2), strides=(2, 2)),
        DropoutLayer(rate=0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
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
model.save('models/finaldiya2.keras', overwrite=True)
model.save('models/finaldiya2.h5')
