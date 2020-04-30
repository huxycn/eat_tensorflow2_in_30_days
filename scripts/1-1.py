import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers


def pre_processing(df_data):
    """
    数据预处理
    :param df_data:
    :return:
    """
    df_pclass = pd.get_dummies(df_data['Pclass'], prefix='Pclass')

    df_sex = pd.get_dummies(df_data['Sex'], prefix='Sex')
    df_result = df_pclass.join(df_sex)

    df_result['Age'] = df_data['Age'].fillna(0)
    df_result['Age_null'] = pd.isna(df_data['Age']).astype('int32')

    df_result['SibSp'] = df_data['SibSp']
    df_result['Parch'] = df_data['Parch']
    df_result['Fare'] = df_data['Fare']

    df_result['Cabin_null'] = pd.isna(df_data['Cabin']).astype('int32')

    df_embarked = pd.get_dummies(df_data['Embarked'], dummy_na=True, prefix='Embarked')

    df_result = df_result.join(df_embarked)

    return df_result


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


def main():
    # 准备数据

    df_train_raw = pd.read_csv('../data/titanic/train.csv')
    df_test_raw = pd.read_csv('../data/titanic/test.csv')

    x_train = pre_processing(df_train_raw)
    y_train = df_train_raw['Survived'].values

    x_test = pre_processing(df_test_raw)
    y_test = df_test_raw['Survived'].values

    print('x_train.shape =', x_train.shape)
    print('x_test.shape =', x_test.shape)

    # 定义模型

    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=(15, )))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # 训练模型

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=30,
                        validation_split=0.2)

    # 评估模型

    plot_metric(history, 'loss')
    plot_metric(history, 'AUC')

    test_loss, test_auc = model.evaluate(x_test, y_test)
    print('Test loss = {}, Test auc = {}'.format(test_loss, test_auc))

    # 使用模型

    prediction = model.predict(x_test[:10])
    print('Pre\tGT')
    for i in range(10):
        print('{:.4}\t{}'.format(prediction[i][0], y_test[i]))

    # 保存模型

    from datetime import datetime
    current_datetime = datetime.strftime(datetime.now(), '%Y%m%d%H%M')
    ckpt_save_path = './checkpoints/weights/titanic-dense-{}.ckpt'.format(current_datetime)
    model.save_weights(ckpt_save_path, save_format='tf')

    model.save('./checkpoints/model', save_format='tf')
    print('save model.')

    print('test saved model')
    model_loaded = tf.keras.models.load_model('./checkpoints/model')
    print(model_loaded.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()
