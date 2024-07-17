import torch
import torch.nn as nn
import torch.nn.functional as F
from pvt_v2 import pvt_v2_b2_li,pvt_v2_b2
#from pvt_v2 import *
import copy
import numpy as np
from collections import OrderedDict
from mmcv.ops import DeformConv2dPack as DCN
from deform_conv_v3 import *
from DESD import *
from res2net_v1b import res2net50_v1b
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class GConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, num=16,dilation=1, bias=False):
        super(GConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, groups=num,padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class FFC(nn.Module):
    def __init__(self, in_depth, AF='prelu'):
        super().__init__()

        # Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth // 2 if in_depth >= 2 else self.in_depth

        # Layers
        self.AF1 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.AF2 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.inConv = nn.Sequential(nn.Conv2d(self.in_depth, self.inter_depth, 1),
                                    nn.BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = nn.Sequential(nn.Conv2d(self.inter_depth, self.inter_depth, 1),
                                     nn.BatchNorm2d(self.inter_depth),
                                     self.AF2)
        self.outConv = nn.Conv2d(self.inter_depth, self.in_depth, 1)

    def forward(self, x):
        x = self.inConv(x)
        _, _, H, W = x.shape
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)
        real_rfft = torch.real(rfft)
        imag_rfft = torch.imag(rfft)
        cat_rfft = torch.cat((real_rfft, imag_rfft), dim=-1)
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1] // 2
        real_rfft = cat_rfft[..., :mid]
        imag_rfft = cat_rfft[..., mid:]
        rfft = torch.complex(real_rfft, imag_rfft)
        spect = torch.fft.irfft2(rfft,s=(H, W))
        out = self.outConv(spect + skip)
        return out

class DSC(nn.Module):#Depthwise_Separable_Convolution
    def __init__(self, in_channel, out_channel, ksize=3,padding=1,bais=True):
        super(DSC, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,groups=in_channel,kernel_size=ksize,padding=padding,bias=bais)
        self.bn=nn.BatchNorm2d(in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,padding=0,bias=bais)
    def forward(self, x):
        out = self.depthwiseConv(x)
        out=self.bn(out)
        out=self.relu(out)
        out = self.pointwiseConv(out)
        return out
        
class DCNN(nn.Module):
    def __init__(self, in_channels, out_channels,num=16):
        super(DCNN,self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1),#,groups=num
            nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1),#,groups=num
            nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True), 
            )
        #self.swish=Swish()
    def forward(self,x):
        x=self.d_conv(x)
        return x
        
class BFCA(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1, num=16, bias=False):
        super(BFCA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes*2, inplanes, 1)
        self.conv2 = nn.Conv2d(inplanes, inplanes, 1)
        
        #self.GrouPconv1=nn.Sequential(nn.Conv2d(inplanes, inplanes,kernel_size=3,padding=1,groups=num),nn.GroupNorm(num_groups=num, num_channels=inplanes),nn.ReLU(inplace=True))      
        self.GrouPconv1=nn.Sequential(nn.Conv2d(inplanes, inplanes,kernel_size=3,padding=1,groups=num),nn.BatchNorm2d(inplanes),nn.ReLU(inplace=True))
        #self.GrouPconv1=nn.Conv2d(inplanes, inplanes,kernel_size=3,padding=1,groups=num)
        self.GrouPconv2=nn.Sequential(nn.Conv2d(inplanes, inplanes,kernel_size=3,padding=1,groups=num),nn.BatchNorm2d(inplanes),nn.ReLU(inplace=True))
        
        self.ADC1=ADeformConv2d(inplanes, inplanes)        
        self.ADC2=ADeformConv2d(inplanes, inplanes)        
        
        self.sharp1=Sharp_kernel(inplanes)
        self.sharp2=Sharp_kernel(inplanes)
        
        self.bn=nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        #self.deform1 = DCN(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        #self.deform2 = DCN(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        
        self.ffc=FFC(inplanes//1)
        self.ffc2=FFC(inplanes//1)
        self.lffn=DCNN(inplanes,inplanes)

    def forward(self, x,y):  
     
        cat= torch.cat([x,y], axis=1)#branch1
        cat=self.conv1(cat)
        gp1=self.GrouPconv1(cat)
        adc1=self.ADC1(gp1)
        cc1=gp1+adc1
        
        cc2=self.ffc(cc1)
        cx1=(cc1+cc2)
        
        add=x+y          #branch2
        gp2=self.GrouPconv2(add)
        adc2=self.ADC2(gp2)
        cy1=gp2+adc2
        
        cy2=self.ffc2(cy1)
        cx2=(cy1+cy2)
        
        cx=cx1+cx2
        cxm=self.lffn(cx)
        out=cx+cxm

        return out
def kernel(num):
    kk= np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    w = np.expand_dims(kk, axis=0)
    w = np.expand_dims(w, axis=0)
    w = np.repeat(w, num, axis=0)
    tensorw = torch.from_numpy(w).float()
    return tensorw
    
class Sharp_kernel(nn.Module):
    def __init__(self, in_chs):
        super(Sharp_kernel, self).__init__()
        self.weight=kernel(in_chs).cuda()
        self.in_chs=in_chs
    def forward(self, x):
        out=F.conv2d(x,self.weight,bias=None,stride=1,padding='same',groups=self.in_chs)
        return out

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
        
class up_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        x = self.up(x)
        #x = self.conv(x)

        return x
 
class senet(nn.Module):
    def __init__(self, in_channels):
        super(senet,self).__init__()
        
        #self.globalpool= F.adaptive_avg_pool2d(xx,(1,1))
        self.line1=torch.nn.Linear(in_channels,int(in_channels//16))
        self.relu=nn.ReLU(inplace=True)
        self.line2=torch.nn.Linear(int(in_channels//16),in_channels)
        self.sigmoid=nn.Sigmoid()
        #self.reshape=torch.reshape()
        
    def forward(self, x):
        #print(x.shape)
        glb=F.adaptive_avg_pool2d(x,(1,1))

        glb=torch.squeeze(glb)
        #print(glb.shape)
        line1=self.line1(glb)
        relu=self.relu(line1)
        exc=self.line2(relu)
        #print(exc.shape)
        sigmoid=self.sigmoid(exc)
        exc=sigmoid.unsqueeze(-1)
        exc=exc.unsqueeze(-1)
        #print(exc.shape)
        #out=torch.mul(x,exc)
        return exc#out
class DBA(nn.Module):
    def __init__(self, in_channels):
        super(DBA,self).__init__()
        
        self.GrouPconv=nn.Sequential(nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1,groups=8),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))
        self.xcon2= nn.Conv2d(in_channels,in_channels,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(2,1,kernel_size=3,padding=1)
        self.sigmoid=nn.Sigmoid()
        self.dsc=DSC(in_channels,in_channels)
        self.in_channels=in_channels
        self.senet1=senet(in_channels//1)
        self.senet2=senet(in_channels//1)
        self.dsc=DSC(in_channels,in_channels)
        self.relu=nn.ReLU(inplace=True)
                               
    def forward(self, x):
        x=self.GrouPconv(x)   
        maxx,e=torch.max(x,dim=1)
        maxx=maxx.unsqueeze(1)
        softm1=self.sigmoid(maxx)

        out1=torch.mul(x,softm1)
        snt1=self.senet1(out1)
        out1=torch.mul(out1,snt1)        
                                
        meann=torch.mean(x,dim=1)
        meann=meann.unsqueeze(1)
        softm2=self.sigmoid(meann)

        out2=torch.mul(x,softm2)
        snt2=self.senet2(out2)
        out2=torch.mul(out2,snt2)                
        out=out1+out2
        return out 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CS(nn.Module):
    def __init__(self, in_channels):
        super(CS,self).__init__()
        
        self.GrouPconv1=nn.Sequential(nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))
        self.GrouPconv2=nn.Sequential(nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))        
        self.ca=ChannelAttention(in_channels)
        self.sa=SpatialAttention()                       
    def forward(self, x):
        x=self.GrouPconv1(x)   
        x=self.GrouPconv2(x)  
        return x        
                             
class ATT(nn.Module):
    def __init__(self, in_channels):
        super(ATT,self).__init__()
        self.xcon2= nn.Conv2d(in_channels,1,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(1,1,kernel_size=3,padding=1)
        self.xcon5 = nn.Conv2d(1,1,kernel_size=5,padding=2)
        self.xcon7 = nn.Conv2d(1,1,kernel_size=7,padding=3)
        self.sigmoid=nn.Sigmoid()
        self.senet=senet(in_channels)
    def forward(self, x):
        sent=self.senet(x)
        x1=self.xcon2(x)
        x3=self.xcon3(x1)
        x5=self.xcon5(x1)
        x7=self.xcon7(x1)
        add1=torch.add(x3,x5)
        add2=torch.add(add1,x7)
        softm=self.sigmoid(add2)
        out1=torch.mul(softm,x)
        out=torch.mul(out1,sent)
        return out
class Spation(nn.Module):
    def __init__(self, in_channels):
        super(Spation,self).__init__()
        self.xcon2= nn.Conv2d(in_channels,1,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(2,1,kernel_size=3,padding=1)
        self.sigmoid=nn.Sigmoid()
        self.dsc=DSC(in_channels,in_channels)
    def forward(self, x):
        x=self.dsc(x) 
        maxx,e=torch.max(x,dim=1)
        meann=torch.mean(x,dim=1)
        maxx=maxx.unsqueeze(1)
        meann=meann.unsqueeze(1)
        x1=self.xcon2(x)
        xc= torch.cat([maxx,meann],dim=1)
        xx=self.xcon3(xc)

        softm=self.sigmoid(xx)
        out=torch.mul(x,softm)
        out=x+out 
       
        return out        
def kernel2(num):
    kk= np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    w = np.expand_dims(kk, axis=0)
    w = np.expand_dims(w, axis=0)
    w = np.repeat(w, num, axis=0)
    tensorw = torch.from_numpy(w).float()
    return tensorw
    
class Sharp_kernel(nn.Module):
    def __init__(self, in_chs):
        super(Sharp_kernel, self).__init__()
        self.weight=kernel2(in_chs).cuda()
        self.in_chs=in_chs
    def forward(self, x):
        out=F.conv2d(x,self.weight,bias=None,stride=1,padding='same',groups=self.in_chs)
        return out
class DSC(nn.Module):#Depthwise_Separable_Convolution
    def __init__(self, in_channel, out_channel, ksize=3,padding=1,bais=True):
        super(DSC, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,groups=in_channel,kernel_size=ksize,padding=padding,bias=bais)
        self.bn=nn.BatchNorm2d(in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,padding=0,bias=bais)
    def forward(self, x):
        out = self.depthwiseConv(x)
        out=self.bn(out)
        out=self.relu(out)
        out = self.pointwiseConv(out)
        return out 
class Boundary_enhance(nn.Module):
    def __init__(self, in_chs):
        super(Boundary_enhance, self).__init__()
        self.sharp=Sharp_kernel(in_chs)
        self.conv=nn.Conv2d(in_chs, 1, kernel_size=1, bias=False)
        self.conv1=nn.Conv2d(in_chs,in_chs, kernel_size=1, bias=False)        
        self.sigmoid=nn.Sigmoid()
        self.dsc=GConvBNR(in_chs,in_chs)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
    
        out=self.sharp(x)
        #out=self.conv1(out)        
        xx=out+x
        xx=self.relu(xx)
        #y=self.conv(xx)
        #y=self.sigmoid(y)
        #xy=xx*y
        #out=xy+xx
        out=self.dsc(xx)
        return out#        
                
class HDCAN(nn.Module):
    def __init__(self, n_classes=1,num_classes=7):
        super(HDCAN, self).__init__()
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'#res2net50_v1b.pth
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.backbone2 = res2net50_v1b()
        path = './res2net50_v1b.pth'#res2net50_v1b.pth
        save_model = torch.load(path)
        model_dict = self.backbone2.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone2.load_state_dict(model_dict)
        
        
        self.be1=CS(64)#Boundary_enhance(64)          
        self.be2=CS(64)#Boundary_enhance(64)  
        self.be3=CS(64)#Boundary_enhance(64)  
        self.be4=CS(64)#Boundary_enhance(64)  
                                
        self.c1 = nn.Conv2d(64, 64, 1, bias=False)#, bias=False
        self.c2 = nn.Conv2d(128,64, 1, bias=False)
        self.c3 = nn.Conv2d(320,64, 1, bias=False)
        self.c4 = nn.Conv2d(512, 64, 1, bias=False)
        
        self.z1 = nn.Conv2d(256, 64, 1, bias=False)        
        self.z2 = nn.Conv2d(512, 64, 1, bias=False)
        self.z3 = nn.Conv2d(1024, 64, 1, bias=False)
        self.z4 = nn.Conv2d(2048, 64, 1, bias=False)
        
        self.cmfa1=BFCA(64)
        self.cmfa2=BFCA(64)
        self.cmfa3=BFCA(64)
        self.cmfa4=BFCA(64)
        
        self.up1 = up_block(64, 64)
        self.up2 = up_block(64, 64)
        self.up3 = up_block(64, 64)
        
        self.dces1=DESD(64,sr_ratio=2)
        self.dces2=DESD(64,sr_ratio=4)
        self.dces3=DESD(64,sr_ratio=8)
        
        self.out1 = nn.Conv2d(64, n_classes, 1)
        self.out2 = nn.Conv2d(64, n_classes, 1)
        self.out3 = nn.Conv2d(64, n_classes, 1)
        self.att1=CS(64)
        self.att2=CS(64)
        self.att3=CS(64)
                
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.sharp=Sharp_kernel(64)
        self.sharp2=Sharp_kernel(64)        
        self.sharp3=Sharp_kernel(64)        
              
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
    
        # if grayscale input, convert to 3 channels
        x0=x
        if x.size()[1] == 1:
            x = self.conv(x)

        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        f1 = self.c1(x1)
        f2 = self.c2(x2)
        f3 = self.c3(x3)
        f4 = self.c4(x4)
        
        res2net = self.backbone2(x)
        y1 = res2net[0]
        y2 = res2net[1]
        y3 = res2net[2]
        y4 = res2net[3]
        
        y1=self.z1(y1)
        y2=self.z2(y2)
        y3=self.z3(y3)        
        y4=self.z4(y4)

        cmfa1= self.cmfa1(f1,y1)
        
        cmfa2= self.cmfa2(f2,y2)
                
        cmfa3= self.cmfa3(f3,y3)
               
        cmfa4= self.cmfa4(f4,y4)
       
        d1 = self.up1(cmfa4)
        dces1=self.dces1(d1,cmfa3)
        
        d2 = self.up2(dces1)
        dces2=self.dces2(d2,cmfa2)
        
        d3 = self.up3(dces2)
        dces3=self.dces3(d3,cmfa1)
                
        p1 = F.interpolate(dces3, scale_factor=4, mode='bilinear')
        p2 = F.interpolate(dces2, scale_factor=8, mode='bilinear')
        p3 = F.interpolate(dces1, scale_factor=16, mode='bilinear')
        
        #p1=self.att1(p1)
        pre1= self.out1(p1)
        #p2=self.att2(p2)        
        pre2= self.out2(p2)
        #p3=self.att3(p3)        
        pre3= self.out3(p3)
        
        pre=pre1+pre2+pre3
        pre=self.sigmoid(pre)
        return pre





