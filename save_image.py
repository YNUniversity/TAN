import torch
import os
# import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataloaders import loaders
from model import chage
from PIL import Image


def main_worker():
    device = torch.device("cuda:0")
    model = chage.DC()
    model.load_state_dict(torch.load('model_result/model_DC_1pixel_30.pt'))
    model.to(device)

    test_data = loaders.AUGAN_scene(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    err1 = []
    err2 = []

    for sample, mask, target, img_name in test_dataloader:
        interation += 1
        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()
       
        with torch.no_grad():
            pre = model(sample, mask)

        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        # print(test.shape)
        # pw-net用的
        # test = test[0:1, :].squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        im2 = im1*255
        predict1 = Image.fromarray(im2.astype(np.uint8))

        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        # pw-net用的
        # test1 = test1[0:1, :].squeeze(0)
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = test1.cpu().numpy()*255
        images = Image.fromarray(image.astype(np.uint8))

        rmse1 = np.sqrt(np.mean((im - im1) ** 2))
        err1.append(rmse1)

        nmse1 = np.mean((im - im1) ** 2)/np.mean((0 - im) ** 2)
        err2.append(nmse1)

        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}_target.png'))

        predict1.save(os.path.join("image_result", f'{image_name}_predict1.png'))
        print(f'saving to {os.path.join("image_result", image_name)}', "RMSE:", rmse1, "NMSE:", nmse1)

        # save image
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)

    print('一阶段测试集均方根误差：', rmse_err)
    print('一阶段测试集归一化均方误差：', nmse_err)


if __name__ == '__main__':
    main_worker()
