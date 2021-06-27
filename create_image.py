from scipy.misc import imsave
import numpy as np

# 解压缩，返回解压后的字典
def unpickle(file):
    import pickle as pk
    fo = open(file, 'rb')
    dict = pk.load(fo, encoding='bytes')    # 为了将ASCII格式转换成字节类型，需要加上encoding='bytes'
    fo.close()
    return dict

# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
for j in range(1, 6):
    dataName = "./data/cifar-10-batches-py/data_batch_" + str(j)   # 读取当前目录下的data_batch12345文件
    Xtr = unpickle(dataName)
    print(dataName + "is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))   # Xtr['data']为图片二进制数据,这边记得前面要加b
        img = img.transpose(1, 2, 0)    # 读取image
        picName = 'train/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'
        imsave(picName, img)
    print(dataName + 'loaded.')

print("test_batch is loading...")

# 生成测试集图片
testXtr = unpickle("./data/cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
    imsave(picName, img)
print("test_batch_loaded.")

