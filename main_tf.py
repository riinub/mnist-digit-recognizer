import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# print(x_train.shape) #(60,000, 28, 28) 60,000 images 28x28 px each
# print(y_train.shape) #60,000 labels one digit per image
# print(x_test.shape) #10,000 images, 28x28 px each

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
# model.summary()

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=5, 
    batch_size = 32, 
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("test accuracy:  ", test_accuracy)

# predictions = model.predict(x_test, verbose=0)
# y_pred = np.argmax(predictions, axis=1)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# wrong = np.where(y_pred != y_test)[0]

# print("Number of mistakes:", len(wrong))

# i = wrong[5]
# plt.imshow(x_test[i], cmap="gray")
# plt.title(f"True: {y_test[i]}, Predicted: {y_pred[i]}")
# plt.axis("off")
# plt.show()


# Get the model's predictions
probabilities = model.predict(x_test, verbose=0)

# Convert probabilities into predicted digit labels
y_pred = np.argmax(probabilities, axis=1)

# Find indices where predictions are incorrect
wrong_indices = np.where(y_pred != y_test)[0]

print("Total test images:", len(y_test))
print("Incorrect predictions:", len(wrong_indices))
print("Test accuracy:", np.mean(y_pred == y_test))

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=range(10),
    columns=range(10)
)

print(cm_df)

# Extract all trainable weights and biases from TensorFlow
weights = model.get_weights()

# Save them in a NumPy .npz file
np.savez(
    "mnist_parameters2.npz",
    **{f"param_{i}": w for i, w in enumerate(weights)}
)

print("Parameters saved successfully!")
