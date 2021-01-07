import tensorflow as tf
import os

#获取图片名字列表，图片流列表
def get_images():
    #图片名字列表
    images_name = []
    #图片流列表
    images_list = []
    #输入地址
    path = input("image path: ")
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1]==".jpg":
                    # 原始图片
                    image_raw = tf.io.read_file(root+"\\"+file)
                    # 原始图片转换为tensor类型
                    image_tensor = tf.image.decode_image(image_raw, channels=3)
                    # 修改尺寸
                    image_final = tf.image.resize(image_tensor, [192, 192])
                    # 绝对色彩信息
                    image_final = image_final / 255.0
                    #升维
                    image_final = image_final[tf.newaxis, ...]
                    #添加到图片名字列表
                    images_name.append(root+"\\"+file)
                    #添加到图片流列表
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
        #升维
        image_final = image_final[tf.newaxis, ...]
        #添加到图片名字列表
        images_name.append(path)
        #添加到图片流列表
        images_list.append(image_final)
    #获取图片名字列表，图片流列表
    return images_name, images_list


#标签名字
labels_name = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
#模型存储位置
checkpoint_save_path = "./planet.ckpt"
#获取已存在的MobileNetV2模型
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
#MobileNet的权重为不可训练
mobile_net.trainable=False
#模型层次：mobile_net，平均池，全连接
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation = 'softmax')])
#判断是否已有模型
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    #加载现有模型
    model.load_weights(checkpoint_save_path)


#获取图片名字列表，图片流列表
images_name, images_stream = get_images()
#测试所有图片
for i in range(len(images_name)):
    #换行
    print()
    # 测试
    result = model.predict(images_stream[i])
    #输出最大概率的标签
    print(images_name[i]+"\nmaximum probability: "+labels_name[int(result.argmax())])
    #输出标签所有概率
    for j in range(len(labels_name)):
        print(labels_name[j] + ": " + str(round(result[0][j], 2)))