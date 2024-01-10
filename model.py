import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf

dir_path = "./data/"
os.listdir(dir_path)

for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

train_images_path = './data/train/'
test_images_path = './data/test/'

labels_path = './data/labels.csv'
labels_df = pd.read_csv(labels_path)
print(labels_df.head())
print(labels_df.describe())


def is_equal_images(target_dir, target_df):
    len_target_dir = len(os.listdir(target_dir))
    len_target_df = len(target_df)
    if len_target_dir == len_target_df:
        print(f"Both are having same number of images:{len_target_dir}")
    else:
        print(f"Target dir having {len_target_dir} images while Target DF having {len_target_df}")


is_equal_images(target_dir=train_images_path, target_df=labels_df)

labels_df['breed'].value_counts().plot.bar(figsize=(20, 10))
print(f"Total number of breeds:{len(labels_df['breed'].unique())}")
print(f"Average Images per breed:{int(labels_df['breed'].value_counts().sum() / len(labels_df['breed'].unique()))}")

# 图片文件名称
filenames = [train_images_path + fname + '.jpg' for fname in labels_df['id']]
# 类别名称
class_names = labels_df['breed'].unique()
# 标签列
target_labels = [breed for breed in labels_df['breed']]

print(len(class_names))
print(len(target_labels))

target_labels_encoded = [label == np.array(class_names) for label in target_labels]

target_labels_encoded[0].astype(int)

from sklearn.model_selection import train_test_split

NUM_IMAGES = len(target_labels)

X_train, X_val, Y_train, Y_val = train_test_split(filenames[:NUM_IMAGES], target_labels_encoded[:NUM_IMAGES],
                                                  test_size=0.2, random_state=42)

print(len(X_train), len(X_val), len(Y_train), len(Y_val))
# 验证：
print(class_names[Y_train[0].argmax()])
print(Y_train[0])

IMAGE_SIZE = 256


# 重塑image 并且返回张量
def process_image(image_path):
    # 读取图片
    img = tf.io.read_file(image_path)
    # 将图像转化为3通道RGB的数值张量
    img = tf.io.decode_image(img, channels=3)
    # 归一化
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize
    # 用最近邻插值法将图像缩放为指定尺寸
    img = tf.image.resize_with_pad(img, 256, 256, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


# 读取图片并处理
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label


print(get_image_label(X_train[0], Y_train[0]))

BATCH_SIZE = 32


def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    if test_data:
        print("Creating Test data")
        test_data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        test_data = test_data.map(process_image).batch(BATCH_SIZE)
        return test_data

        # 创建验证集数据
    if valid_data:
        print("Creating Validation data")
        valid_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        valid_data = valid_data.map(get_image_label).batch(BATCH_SIZE)
        return valid_data

    # 重新打乱并且创建训练数据
    else:
        print("Creating Training Data")
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))).shuffle(buffer_size=len(X))
        # 处理图像尺寸
        train_data = train_data.map(get_image_label).batch(BATCH_SIZE)
        return train_data
    # 训练集


train_data = create_data_batches(X_train, Y_train)
# 验证集合
valid_data = create_data_batches(X_val, Y_val, valid_data=True)

from tensorflow.python.keras import layers


# 创建MobileNet V2模型
def create_model():
    # MobileNetV2 作为基础模型不使用模型的全连接层（后续自己再添加一个全连接层） 模型的输出为所有类别
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                classes=len(class_names))
    # 冻结原模型中的一些参数，防止在后续的训练轮次中破坏这些参数信息
    base_model.trainable = False

    # 创建输入层
    inputs = layers.Input(shape=(256, 256, 3))
    # 将输入传递给基础模型
    x = base_model(inputs, training=False)
    # 添加全局平均池化层，将每个特征图的平均值作为输出。可以减少模型参数数量
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    # 添加一个20%的Dropout层 防止过拟合
    x = layers.Dropout(0.3)(x)
    # 手动添加一个全连接层，输出所有类别的概率分布，激活函数用softmax
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    ModelDogBreed = tf.keras.Model(inputs, outputs)
    # 损失函数为分类交叉熵 优化器选择Adam 用accuracy作为评估指标
    ModelDogBreed.compile(loss="categorical_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["accuracy"])

    return ModelDogBreed


model = create_model()
# 定义一个早停 如果验证集上连续两个epoch都没有显著改善，就早停
EarlyStoppingCallbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, baseline=None, restore_best_weights=True
)
print(model.summary())

# 模型拟合
ModelDogBreed_History = model.fit(train_data,
                                  steps_per_epoch=len(train_data),
                                  epochs=10,
                                  validation_data=valid_data,
                                  validation_steps=len(valid_data),
                                  callbacks=[EarlyStoppingCallbacks]
                                  )


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


plot_loss_curves(ModelDogBreed_History)

predictions = model.predict(valid_data)
print(predictions.shape)


# 将预测概率中最大值所对应的标签提取出来
def get_pred_label(prediction_probabilities):
    return class_names[np.argmax(prediction_probabilities)]


Y_val_coded = Y_val[0].astype(int)
print(f"The true beer is {class_names[np.argmax(Y_val_coded)]}")
print(f"The predict beer is {get_pred_label(predictions[0])}")


def unbatchify(data):
    images = []
    labels = []
    # 循环处理未分批数据
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(class_names[np.argmax(label)])
    return images, labels


# 取消验证集的批处理
val_images, val_labels = unbatchify(valid_data)
val_images[0], val_labels[0]


def plot_pred(prediction_probabilities, labels, images, n=1):
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    # 获得预测标签
    pred_label = get_pred_label(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    # 正确绿色错误红色
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
    # 括号内是真实标签
    plt.title("{} {:2.0f}% ({})".format(pred_label,
                                        np.max(pred_prob) * 100,
                                        true_label),
              color=color)


def plot_pred_conf(prediction_probabilities, labels, n=1):
    pred_prob, true_label = prediction_probabilities[n], labels[n]

    # 获得预测标签
    pred_label = get_pred_label(pred_prob)

    # 找到前十高的种类下标
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    # 获得概率值
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # 获得标签值
    top_10_pred_labels = class_names[top_10_pred_indexes]

    top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                       top_10_pred_values,
                       color="grey")
    plt.xticks(np.arange(len(top_10_pred_labels)),
               labels=top_10_pred_labels,
               rotation="vertical")

    # 真实的为绿色
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
    else:
        pass


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)


def MyPic_plot_pred(prediction_probabilities, images, prediction_label):
    pred_prob, image, pred_label = prediction_probabilities, images, prediction_label

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    color = "deeppink"

    # 括号内是真实标签
    plt.title("{} {:2.0f}% ({})".format(pred_label,
                                        np.max(pred_prob) * 100,
                                        pred_label),
              color=color)


def MyPic_plot_pred_conf(prediction_probabilities, prediction_label):
    prediction_probabilities = prediction_probabilities.flatten()
    print(prediction_probabilities.shape)
    top_10_indexes = np.argsort(prediction_probabilities)[-10:][::-1]
    top_10_features = class_names[top_10_indexes]
    top_10_probabilities = prediction_probabilities[top_10_indexes]
    top_plot = plt.bar(top_10_features, top_10_probabilities, color="pink")
    plt.xlabel("Feature Name")
    plt.ylabel("Probability")
    plt.title("Top 10 Features with Highest Probabilities")
    plt.xticks(np.arange(len(top_10_features)),
               labels=top_10_features,
               rotation="vertical")


# 创建一个函数用于预测自己输入的图片
def predict_my_pic(img_path):
    img = tf.io.read_file(img_path)
    # 将图像转化为3通道RGB的数值张量
    img = tf.io.decode_image(img, channels=3)
    # 归一化
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize
    # 用最近邻插值法将图像缩放为指定尺寸
    img = tf.image.resize_with_pad(img, 256, 256, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 将图像添加到它是唯一成员的批中。
    img = (np.expand_dims(img, 0))
    # 获得预测概率
    prediction = model.predict(img)
    # print(prediction.shape)
    # 获得概率最大的类别
    prediction_label = get_pred_label(prediction)
    return prediction, prediction_label


def op():
    my_pic_path = r'C:\Users\caipengcheng\Desktop\深度学习\期末\my_picture\萨摩耶.jpg'
    prediction, pre_label = predict_my_pic(my_pic_path)
    image = tf.io.read_file(my_pic_path)
    image = tf.io.decode_image(image, channels=3)
    # 可视化预测结果
    num_rows = 5
    num_cols = 2
    num_images = num_rows * num_cols
    plt.figure(figsize=(5 * 2 * num_cols, 5 * num_rows))
    plt.subplot(num_rows, 2 * num_cols, 3)
    MyPic_plot_pred(prediction_probabilities=prediction,
                    images=image,
                    prediction_label=pre_label)
    plt.subplot(num_rows, 2 * num_cols, 4)
    MyPic_plot_pred_conf(prediction_probabilities=prediction,
                         prediction_label=pre_label)
    plt.tight_layout(h_pad=1.0)
    plt.show()


import streamlit as st


def main():
    # Streamlit App 的主要部分
    st.title('深度学习模型预测可视化')

    # 上传图片文件
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = tf.io.decode_image(uploaded_file.read(), channels=3)
        st.image(image.numpy(), caption="上传的图片", use_column_width=True)

        # 进行预测
        prediction, pre_label = predict_my_pic(uploaded_file)

        # 可视化预测结果
        st.subheader("预测结果可视化")
        MyPic_plot_pred(prediction_probabilities=prediction,
                        images=image,
                        prediction_label=pre_label)

        st.subheader("预测置信度可视化")
        MyPic_plot_pred_conf(prediction_probabilities=prediction,
                             prediction_label=pre_label)


if __name__ == '__main__':
    main()
