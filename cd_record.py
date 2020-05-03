import tensorflow as tf
import os
import matplotlib.pyplot as plt
#from keras.callbacks import ModelCheckpoint

num_epochs = 1
batch_size = 32
learning_rate = 0.001
data_dir = 'E:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_file = data_dir + '/train/train.tfrecords'

def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']

if __name__ == '__main__':
    train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
    train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
    train_filenames = train_cat_filenames + train_dog_filenames
    train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)  # 将 cat 类的标签设为0，dog 类的标签设为1

    if not os.path.exists(tfrecord_file):
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for filename, label in zip(train_filenames, train_labels):
                image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                feature = {                             # 建立 tf.train.Feature 字典
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
                writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件

    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件

    feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    train_dataset = raw_dataset.map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    for image, label in train_dataset:
        plt.title('cat' if label == 0 else 'dog')
        plt.imshow(image.numpy())
        plt.show()
        break
    
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint= tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', save_weights_only=True, verbose=1, save_best_only=True, mode='max')
    callbacks_list= [checkpoint]
    '''
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    # 使用tf.train.CheckpointManager管理Checkpoint
    manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
    '''
    
    # Fit the model
    if os.path.exists(filepath):
        model.load_weights(filepath)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks_list)
    '''
    path = manager.save(checkpoint_number=batch_index)
    print("model saved to %s" % path)
    '''
    # 构建测试数据集
    test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
    test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
    test_labels = tf.concat([
        tf.zeros(test_cat_filenames.shape, dtype=tf.int32), 
        tf.ones(test_dog_filenames.shape, dtype=tf.int32)], 
        axis=-1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))