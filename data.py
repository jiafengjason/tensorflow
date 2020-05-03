import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label

'''
for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()
    break
    
'''
mnist_dataset_rot90 = mnist_dataset.map(map_func=rot90, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for image, label in mnist_dataset_rot90:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()
    break

mnist_dataset_batch = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=10000).batch(4)
for images, labels in mnist_dataset_batch:
    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()
    for i in range(4):
        axs[i].set_title(labels.numpy()[i])
        axs[i].imshow(images.numpy()[i, :, :, 0])
    plt.show()
    break

