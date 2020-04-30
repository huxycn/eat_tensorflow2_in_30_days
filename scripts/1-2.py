import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models


BATCH_SIZE = 100


def load_image(img_path, size=(32, 32)):
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, '.*/automobile/.*') \
        else tf.constant(0, tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size) / 255.0
    return img, label


def main():
    ds_train = tf.data.Dataset.list_files('../data/cifar2/train/*/*.jpg').map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = tf.data.Dataset.list_files('../data/cifar2/test/*/*.jpg').map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # plt.figure(figsize=(9, 9))
    # for i, (img, label) in enumerate(ds_train.unbatch().take(9)):
    #     ax = plt.subplot(3, 3, i+1)
    #     ax.imshow(img.numpy())
    #     ax.set_title('label={}'.format(label))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # plt.show()

    for x, y in ds_train.take(1):
        print(x.shape, y.shape)

    tf.keras.backend.clear_session()

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # 训练
    from datetime import datetime
    log_dir = './log/1-2/{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(ds_train, epochs=10, validation_data=ds_test,
                        callbacks=[tensorboard_callback], workers=4)

    import pandas as pd
    df_history = pd.DataFrame(history.history)
    df_history.index = range(1, len(df_history)+1)
    df_history.index.name = 'epoch'

    print(df_history.head(10))

    def plot_metric(history, metric):
        """
        训练过程loss、auc变化折线图
        :param history:
        :param metric:
        :return:
        """
        train_metrics = history.history[metric]
        val_metrics = history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend(['train_' + metric, 'val_' + metric])
        plt.show()

    # plot_metric(history, 'accuracy')
    # plot_metric(history, 'loss')

    val_loss, val_accuracy = model.evaluate(ds_test, workers=4)
    print(val_loss, val_accuracy)

    prediction = model.predict(ds_test)
    print(prediction)

    for x, y in ds_test.take(1):
        print(model.predict_on_batch(x[:20]))

    model.save_weights('./1-2_checkpoints/weights.ckpt', save_format='tf')
    model.save('./1-2_checkpoints/saved_model_weights', save_format='tf')

    model_loaded = tf.keras.models.load_model('./1-2_checkpoints/saved_model_weights')
    prediction_ = model.loaded.evaluate(ds_test)
    print(prediction_)
    

if __name__ == '__main__':
    main()
    #
    # from tensorboard import notebook
    #
    # notebook.list()
    #
    # notebook.start('--logdir ./log/1-2')

