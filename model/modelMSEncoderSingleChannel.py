import torch.nn as nn
import torch.nn.functional as F
import torch

from model.resnet import resnet34


class EyeMSResNetBlockMSEncoder(nn.Module):
    def __init__(self,reduction=4,keranl=3, zero_init_residual=True,channel=1):
        super(EyeMSResNetBlockMSEncoder, self).__init__()
        self.encoder = resnet34(channel=channel).cuda()
        modules = list(self.encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.msModel=EyeMSResNetBlockMS(reduction=reduction,keranl=keranl, zero_init_residual=zero_init_residual,channel=channel)

    def forward(self,x):
        # uncomment this line if you get size error
        #  x =  nn.functional.interpolate(x, size=224)
        embedding=self.encoder(x)
        out=self.msModel(x,embedding)
        return out


class EyeMSResNetBlockMS(nn.Module):
 def __init__(self, reduction=4,keranl=3, zero_init_residual=True,channel=1):
    super(EyeMSResNetBlockMS, self).__init__()
    self.dim=512
    self.in_channel=channel

    self.block0=BasicBlock(inplanes=32,size=4,planes=512 // reduction,in_channel=self.in_channel)

    self.block1=BasicBlock(inplanes=512 // reduction,size=8,planes=512 // reduction,in_channel=self.in_channel)

    self.block2=BasicBlock(inplanes=512 // reduction,size=16,planes=512 // reduction,in_channel=self.in_channel)
    self.block3=BasicBlock(inplanes=512 // reduction,size=32,planes=512 // reduction,in_channel=self.in_channel)
    self.block4=BasicBlock(inplanes=512 // reduction,size=64,planes=256 // reduction,in_channel=self.in_channel)
    self.block5=BasicBlock(inplanes=256 // reduction,size=128,planes=256 // reduction,in_channel=self.in_channel)
    self.block6=BasicBlock(inplanes=256 // reduction,size=256,planes=128 // reduction,in_channel=self.in_channel)
    self.conv3=nn.Conv2d(128 // reduction, 1, kernel_size=keranl, stride=1, padding=1,bias=False)
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual:
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
 def forward(self,x, embedding):
     embedding=torch.reshape(embedding,(embedding.size()[0],32,4,4))

     out=self.block0(embedding,x)
     out=self.block1(out,x)
     out=self.block2(out,x)
     out=self.block3(out,x)
     out=self.block4(out,x)
     out=self.block5(out,x)
     out=self.block6(out,x)

     out=self.conv3(out)
     out=torch.sigmoid(out)

     return out
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=0):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, size=8, in_channel=1):
        super(BasicBlock, self).__init__()
        self.size=size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv0=conv1x1(inplanes + in_channel , planes , stride)
        self.bn0 = norm_layer(planes)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(planes , planes , stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, input_image):
        out1 = nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
        out2 =nn.functional.interpolate(input_image, size=self.size)

        x = torch.cat([out1, out2], dim=1)
        x=self.conv0(x)
        x=self.bn0(x)
        x=self.relu0(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class CRNBlock(nn.Module):
    def __init__(self, in_cahnnel=515, size=4, dim=512, channel=1,keranl=4):
        super(CRNBlock, self).__init__()
        self.size=size

        self.con1= nn.Conv2d(in_cahnnel+channel, dim, kernel_size=(keranl,keranl),stride=1, padding=1,bias=True)
        self.bn1 = nn.LayerNorm([dim ,size,size])

        self.con2= nn.Conv2d(dim, dim, kernel_size=(keranl,keranl),stride=1, padding=1,bias=True)

        self.bn2 = nn.LayerNorm([dim ,size,size]) # nn.BatchNorm2d(dim ) # nn.BatchNorm2d(dim ) #


    def forward(self, net, input_image ):
        out1=nn.functional.interpolate(net,size=self.size,mode='bilinear',align_corners=True)
        out2=nn.functional.interpolate(input_image,size=self.size)

        out=torch.cat([out1,out2], dim=1)

        out=self.con1(out)
        out=self.bn1(out)

        out=lrelu(out)
        out = self.con2(out)
        out = self.bn2(out)

        out = lrelu(out)
        return out