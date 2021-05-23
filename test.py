import tensorflow as tf
import os


# 获取图片名字列表，图片流列表
def get_images():
    # 图片名字列表
    images_name = []
    # 图片流列表
    images_list = []
    # 输入路径
    # path = input("image path: ")
    path = ".\\test_photos"
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == ".jpg":
                    # 原始图片
                    image_raw = tf.io.read_file(root + "\\" + file)
                    # 原始图片转换为tensor类型
                    image_tensor = tf.image.decode_image(image_raw, channels=3)
                    # 修改尺寸
                    image_final = tf.image.resize(image_tensor, [192, 192])
                    # 绝对色彩信息
                    image_final = image_final / 255.0
                    # 升维
                    image_final = image_final[tf.newaxis, ...]
                    # 添加到图片名字列表
                    images_name.append(root + "\\" + file)
                    # 添加到图片流列表
                    images_list.append(image_final)
    else:
        # 原始图片
        image_raw = tf.io.read_file(path)
        # 原始图片转换为tensor类型
        image_tensor = tf.image.decode_image(image_raw, channels=3)
        # 修改尺寸
        image_final = tf.image.resize(image_tensor, [192, 192])
        # 绝对色彩信息
        image_final = image_final / 255.0
        # 升维
        image_final = image_final[tf.newaxis, ...]
        # 添加到图片名字列表
        images_name.append(path)
        # 添加到图片流列表
        images_list.append(image_final)
    # 获取图片名字列表，图片流列表
    return images_name, images_list


# 根据文件夹名字获取标签
def get_label_name(path):
    # 标签左开头
    left = len(path) - 1
    # 标签右结尾
    right = len(path) - 1
    # 是否已遍历过’/‘
    flag = 1
    # 从后往前遍历字符串
    for i in range(len(path) - 1, -1, -1):
        # 是否是‘/’
        if (path[i] == '\\'):
            # 未遍历过’/‘
            if (flag):
                # 标签右结尾
                right = i
                flag = 0
            # 遍历过’/‘
            else:
                # 标签左开头
                left = i + 1
                break
    # 返回标签名字
    return path[left:right]


# 标签名字
labels_name = ['cherry', 'chinese rose', 'daisy', 'dandelion', 'myosotis', 'poppy', 'roses', 'sunflowers', 'tulips',
               'violet']
# 模型存储位置
checkpoint_save_path = "./plant.ckpt"
# 获取已存在的MobileNetV2模型
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# MobileNet的权重为不可训练
mobile_net.trainable = False
# 模型层次：mobile_net，平均池，全连接
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels_name), activation='softmax')])
# 判断是否已有模型
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    # 加载现有模型
    model.load_weights(checkpoint_save_path)

# 获取图片名字列表，图片流列表
images_name, images_stream = get_images()
# 各种类图片标签命中
image_acc = list(0 for i in range(len(labels_name)))
# 各种类图片计数
image_count = list(0 for i in range(len(labels_name)))
# 测试所有图片
for i in range(len(images_name)):
    # 换行
    print()
    # 测试
    result = model.predict(images_stream[i])
    # 实际标签
    real_label = get_label_name(images_name[i])
    # 预测得到的标签
    pre_label = labels_name[int(result.argmax())]

    # 实际标签与预测得到的标签相同
    if (real_label == pre_label):
        image_acc[int(result.argmax())] = image_acc[int(result.argmax())] + 1
        image_count[int(result.argmax())] = image_count[int(result.argmax())] + 1
    # 实际标签与预测得到的标签不相同
    else:
        # 查找图片实际标签，并对图片计数加一
        for i in range(len(labels_name)):
            if (labels_name[i] == real_label):
                image_count[i] = image_count[i] + 1
                break

    # 输出最大概率的标签
    print(images_name[i] + "\nmaximum probability: " + pre_label)
    # 输出标签所有概率
    for j in range(len(labels_name)):
        print(labels_name[j] + ": " + str(round(result[0][j] * 100, 2)) + "%")

# 输出全部图片预测的准确率
print("---------------all images acc---------------")
for i in range(len(labels_name)):
    if (image_count[i]):
        print(labels_name[i] + " acc: " + str(round(image_acc[i] / image_count[i] * 100, 2)) + "%")
    else:
        print(labels_name[i] + " acc: 0.00%")
