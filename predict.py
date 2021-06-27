import torch
import torchvision.transforms as transforms
from resnet import ResNet18
from PIL import Image

def predict_(img):
    data_ransform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std) → mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    img = data_ransform(img)
    img = torch.unsqueeze(img, dim=0)

    model = ResNet18()
    model_weight_pth = './model/net_128.pth'
    model.load_state_dict(torch.load(model_weight_pth))

    model.eval()
    classes = {'0': '飞机', '1': '汽车', '2': '鸟', '3': '猫', '4': '鹿', '5': '狗', '6': '青蛙', '7': '马', '8': '船', '9': '卡车'}
    with torch.no_grad():
        output = torch.squeeze(model(img))
        print(output)
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    return classes[str(predict_cla)], predict[predict_cla].item()

if __name__ == '__main__':
    img = Image.open('./test/0_3.jpg')
    net = predict_(img)
    print(net)