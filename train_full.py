import os
import sys
from tqdm import tqdm
#from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torchvision.utils import make_grid
from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,BCELoss

from dataset import *
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./breastseg/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path


from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from HDCAN  import *

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target,dim=(2,3))
    y_sum = torch.sum(target * target,dim=(2,3))
    z_sum = torch.sum(score * score,dim=(2,3))
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - torch.mean(loss)
    return loss
    
def Tver_loss(score, target):
    target = target.float()
    smooth = 1e-5
    a=0.3
    b=0.7
    intersect = torch.sum(score * target,dim=(2,3))
    y_sum = torch.sum((1-score) * target,dim=(2,3))
    z_sum = torch.sum(score*(1-target),dim=(2,3))
    loss = (intersect + smooth) / (intersect+a*y_sum + b*z_sum + smooth)
    loss = 1-loss
    loss=torch.pow(loss,1/1.5)
    return torch.mean(loss)   

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 1
patch_size = (112, 112, 80)

def dice_m(score, target, smooth=1e-10):

    intersect = torch.sum(score * target,dim=(2,3))
    y_sum = torch.sum(target * target,dim=(2,3))
    #print(torch.sum(y_sum).item())
    z_sum = torch.sum(score * score,dim=(2,3))
    dc= (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return torch.mean(dc.float())
def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x)) / np.std(x)
    return x
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def validate(net, valid_loader,batch_size):
        # validation steps
        with torch.no_grad():
            net.eval()
            valid_dc = 0
            loss = 0
            for images, labels in valid_loader:
                images = images.cuda().float()
                labels = labels.cuda().float()
                pred = net(images)
                #pred=F.softmax(pred,dim=1)
                dc = dice_m(pred,labels)
                valid_dc+=dc
                #print(dc)
                loss += dice_loss(pred,labels)
        return valid_dc/len(valid_loader),loss/len(valid_loader)
def Heatmap(image):
      Va_images=image.reshape(image.shape[1],image.shape[2],image.shape[0])
      heat=np.mean(Va_images,axis=-1)
      heatmap = np.maximum(heat,0)
      heatmap /= np.max(heatmap)
      heatmap = cv2.resize(heatmap,(320,320))
      heatmap = np.uint8(255 * heatmap)    
      heatmap = cv2.applyColorMap(heatmap//1, cv2.COLORMAP_JET)    
      #cv2.imwrite(path_save+str(nm*(fold_test-1)+i+1)+'h.tif',heatmap)   
      return heatmap                
                       
def test_net(net,device,tst_path):
        state_dict = torch.load('./breastseg/best_model.pth')
        net.load_state_dict(state_dict)
        #tst_path='/storage/yyc/data/tst/'
        listname=glob(tst_path+'*.jpg')
        path_save=r'./breastseg/save/HDCAN/'
        isExists=os.path.exists(path_save)
        if not isExists:
           os.makedirs(path_save)
        
        with torch.no_grad():
            net.eval()
            for image_path in listname:
                 image = cv2.imread(image_path)
                 #X =preprocess_input(image)
                 image = np.array(image)
                 image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                 image=preprocess_input(image)
                 image=image.reshape(1,image.shape[0],image.shape[1])
                 image=np.expand_dims(image, axis=0)
                 image=torch.from_numpy(image)
                
                 mg= image.cuda().float()
                 y1 = net(mg)
                 y1=torch.squeeze(y1)
                 #y = F.softmax(y1, dim=1)
                 mm = y1.cpu().data.numpy()
                 mm[mm>=0.5]=1#mmg=y[0,:,:,:]
                 mm[mm<0.5]=0#mg=np.squeeze(np.argmax(mmg,0))
                 cv2.imwrite(path_save+image_path.split('/')[-1].split('.')[0]+'.png',mm*255)
device = torch.device('cuda:0')
if __name__ == "__main__":
    ## make logger file

    def create_model(ema=False):
        # Network definition
        net=HDCAN()
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    db_train = ISBI_Dataset(train_data_path+'train/',True,True)
    db_val = ISBI_Dataset(train_data_path+'val/',True,False)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    batch_size=16
    trainloader=DataLoader(db_train,batch_size,shuffle = True)
    val_loader=DataLoader(db_val,batch_size,shuffle = False)
    print('num:',len(trainloader))    
    model.train()

    optimizer =optim.Adam(model.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    ce_loss =BCELoss()#F.cross_entropy#CrossEntropyLoss()#()#BCEWithLogitsLoss()#

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)
    lr_ = base_lr

    val_dice=0
    val_ls=1
    epochs=100#max_epoch
    #pathsave=''
    for epoch_num in tqdm(range(epochs), ncols=100):
        model.train()
        time1 = time.time()
        lossm=0
        for  i,(image,label) in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = image,label#sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda().float()

            outputs = model(volume_batch)
            #outputs_soft = torch.softmax(outputs, dim=1)
            loss= dice_loss(outputs, label_batch)#+0.5*loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossm+=loss.item()

            iter_num = iter_num + 1
        v_dc,val_loss=validate(model,val_loader,batch_size)
        print('epoch:%d  train loss:%f' % (epochs+1,lossm/len(trainloader)))
        if v_dc >val_dice:
                print('val_dice changed:',v_dc.item(),'model saved, val_loss:',val_loss.item())
                #val_dice=v_dc
                val_dice=v_dc
                torch.save(model.state_dict(), './breastseg/best_model.pth')
    tst_path = r'./breastseg/tst/'
    test_net(model,device,tst_path)
