import os
import tensorflow as tf
import numpy as np
from utils import MNISTLoader

num_epochs = 1
batch_size = 50
learning_rate = 0.001

saveFile = os.path.join('saved_sequential','1')
data_loader = MNISTLoader()
if os.path.exists(saveFile):
    model = tf.saved_model.load(saveFile)
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
    tf.saved_model.save(model, saveFile)