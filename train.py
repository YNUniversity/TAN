from __future__ import print_function, division
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
from lib import loaders
from model import chage
import torch.nn as nn
import warnings

if __name__ == '__main__':
    print('GPU:', torch.cuda.device_count())
    device = torch.device("cuda:0")

    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    Radio_train = loaders.AUGAN_scene(phase="train")
    Radio_val = loaders.AUGAN_scene(phase="val")
    Radio_test = loaders.AUGAN_scene(phase="test")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    batch_size = 4

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    }

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    cudnn_enabled = torch.backends.cudnn.enabled
    model = chage.DC()
    model.cuda()
    

    def calc_loss_dense(pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)  
        # print(target.size())
        return loss


    def print_metrics(metrics, epoch_samples, phase):
        outputs1 = []
        for k in metrics.keys():
            outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs1)))

    def train_model(model, optimizer, scheduler, num_epochs=25, targetType="dense"):
        best_model_wts = copy.deepcopy(model.state_dict()) 
        best_loss = 1e10  

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()  
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step() 
                    for param_group in optimizer.param_groups:
                        print("learning rate", param_group['lr'])
                    model.train()
                else:
                    model.eval()

                metrics = defaultdict(float)  
                epoch_samples = 0  
                if targetType == "dense":
                    for sample, mask, target, name in tqdm(dataloaders[phase]):
                        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):  
                            outputs = model(sample, mask)
                            loss = calc_loss_dense(outputs, target, metrics)
                            # print(loss)  

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        epoch_samples += target.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        model.load_state_dict(best_model_wts)
        return model  

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler)

    try:
        os.mkdir('model_result')
    except OSError as error:
        print(error)

    torch.save(model.state_dict(), 'model_result/model_DC_1pixel_30.pt')
