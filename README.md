# CFAR10-with-tensorflow
This code is an example of deep learning using TensorFlow for image classification using the CIFAR-10 dataset. Here's an explanation of each part of the code:

1. **Import Statements**: In this section, TensorFlow and other necessary libraries are imported.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
```

2. **Load CIFAR-10 Dataset**: This part of the code downloads and loads the CIFAR-10 dataset from TensorFlow. This dataset consists of 60,000 images with dimensions of 32x32 pixels, categorized into 10 different classes.

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

3. **Normalize the Data**: Input images are normalized by dividing their pixel values by 255, which scales them to a range between 0 and 1.

```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
```

4. **One-Hot Encoding**: Class labels are one-hot encoded. In other words, each label is transformed into a vector of length 10, where all elements are 0 except for the element corresponding to the respective class, which is set to 1.

```python
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

5. **Define the MLP Model**: In this section, the architecture of the deep learning model is defined. This model is a Multi-Layer Perceptron (MLP) with input size 32x32x3 (three color channels). It consists of 3 fully connected (Dense) layers with a specified number of neurons and Dropout layers to prevent overfitting.

```python
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```

6. **Compile the Model**: The defined model is compiled with a choice of loss function (categorical cross-entropy) and an optimization algorithm (Adam). The accuracy metric is also specified.

```python
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.002),
              metrics=['accuracy'])
```

7. **Train the Model**: The model is trained on the training data. The training data is divided into smaller batches, and the model is trained for 10 epochs.

```python
model.fit(x_train, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(x_test, y_test))
```

8. **Evaluate the Model**: After training, the model is evaluated on the test data, and the accuracy of the model on the test data is displayed.

```python
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy:', accuracy)
```

This code creates a deep neural network model with dropout layers for image classification on the CIFAR-10 dataset and trains the model using the Adam optimizer. Finally, it displays the accuracy of the model on the test data.
