import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, preprocessing, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


train_data_path = '../data/imdb/train.csv'
test_data_path = '../data/imdb/test.csv'

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20


def split_line(line):
    arr = tf.strings.split(line, '\t')
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return text, label


def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(
        stripped_html, '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation


@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24*60*60)
    hour = tf.cast(today_ts//3600+8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts%3600)//60, tf.int32)
    second = tf.cast(tf.floor(today_ts%60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format('{}', m)) == 1:
            return tf.strings.format('0{}', m)
        else:
            return tf.strings.format('0{}', m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)],
                                 separator=':')
    tf.print('='*80, end='')
    tf.print(timestring)


class CnnModel(models.Model):

    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv1 = layers.Conv1D(16, kernel_size=5, name='conv_1', activation='relu')
        self.pool1 = layers.MaxPool1D(name='maxpool_1')
        self.conv2 = layers.Conv1D(128, kernel_size=2, name='conv_2', activation='relu')
        self.pool2 = layers.MaxPool1D(name='maxpool_2')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def main():
    ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path])\
        .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .shuffle(buffer_size=1000)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.experimental.AUTOTUNE)


    ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path])\
        .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .shuffle(buffer_size=1000)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    vectorize_layer = TextVectorization(
        standardize=clean_text,
        split='whitespace',
        max_tokens=MAX_WORDS - 1,
        output_mode='int',
        output_sequence_length=MAX_LEN
    )

    ds_text = ds_train_raw.map(lambda text, label: text)
    vectorize_layer.adapt(ds_text)
    print(vectorize_layer.get_vocabulary()[:100])

    ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test_raw.map(lambda text, label: (vectorize_layer(text), label)) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    tf.keras.backend.clear_session()

    model = CnnModel()
    model.build(input_shape=(None, MAX_LEN))
    model.summary()

    optimizer = optimizers.Nadam()
    loss_func = losses.BinaryCrossentropy()
    train_loss = metrics.Mean(name='train_loss')
    train_acc = metrics.BinaryAccuracy(name='train_acc')
    val_loss = metrics.Mean(name='val_loss')
    val_acc = metrics.BinaryAccuracy(name='val_acc')

    @tf.function
    def train_step(model, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(labels, predictions)

    @tf.function
    def val_step(model, features, labels):
        predictions = model(features, training=False)
        loss_ = loss_func(labels, predictions)
        val_loss.update_state(loss_)
        val_acc.update(labels, predictions)

    def train_model(model, ds_train, ds_val, epochs):
        for epoch in tf.range(1, epochs+1):
            for features, labels in ds_train:
                train_step(model, features, labels)

            for features, labels in ds_val:
                val_step(model, features, labels)

            logs = 'Epoch={} | loss:{} - acc:{} - val_loss:{} - val_acc:{}'

            if epoch % 1 == 0:
                printbar()
                tf.print(tf.strings.format(logs, (
                    epoch, train_loss.result(), train_acc.result(), val_loss.result(), val_acc.result()
                )))
                tf.print()
            train_loss.reset_states()
            val_loss.reset_states()
            train_acc.reset_states()
            val_acc.reset_states()

    train_model(model, ds_train, ds_test, epochs=6)


if __name__ == '__main__':
    main()
    # print(tf.test.is_gpu_available())
