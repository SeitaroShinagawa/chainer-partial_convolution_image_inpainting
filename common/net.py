import numpy as np
import sys, os
#sys.path.append(os.path.dirname(__file__))

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Variable

from chainer import cuda
from chainer import serializers
import numpy as np
from chainer import Variable
import chainer
import math
import copy

def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

class ResBlock(chainer.Chain):
    def __init__(self, ch, bn=True, activation=F.relu):
        self.bn = bn
        self.activation = activation
        layers = {}
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1)
        if bn:
            layers['bn0'] = L.BatchNormalization(ch)
            layers['bn1'] = L.BatchNormalization(ch)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x, test):
        h = self.c0(x)
        if self.bn:
            h = self.bn0(h, test=test)
        h = self.activation(h)
        h = self.c1(h)
        if self.bn:
            h = self.bn1(h, test=test)
        return h + x


class PConv(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='none-3', activation=F.relu, dropout=False, noise=False):
        "Assuming biases are None"
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        layers = {}
        #w = chainer.initializers.Normal(0.02)
        w = chainer.initializers.Normal(0.02)
        if sample=='down-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 2, 2, initialW=w)
            layers['m'] = L.Convolution2D(ch0, ch1, 5, 2, 2, initialW=1.0, nobias=True)
        elif sample=='down-7':
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 2, 3, initialW=w)
            layers['m'] = L.Convolution2D(ch0, ch1, 7, 2, 3, initialW=1.0, nobias=True) 
        elif sample=='down-3':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 2, 1, initialW=w)
            layers['m'] = L.Convolution2D(ch0, ch1, 3, 2, 1, initialW=1.0, nobias=True)
        else:
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
            layers['m'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=1.0, nobias=True)
        self.maskW = copy.deepcopy(layers['m'].W.data)
        if bn:
            if self.noise:
                layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['batchnorm'] = L.BatchNormalization(ch1)
        super(PConv, self).__init__(**layers)

    def __call__(self, x, mask):
        #h = self.c(x) - self.b
        self.m.W.data = self.xp.array(self.maskW) #mask windows are set by 1
        h = self.c(x*mask) #(B,C,H,W)
        B,C,H,W = h.shape
        b = F.transpose(F.broadcast_to(self.c.b,(B,H,W,C)),(0,3,1,2))
        h = h - b
        mask_sums = self.m(mask)
        mask_new = (self.xp.sign(mask_sums.data-0.5)+1.0)*0.5
        mask_new_b = mask_new.astype("bool")
        
        mask_sums = F.where(mask_new_b,mask_sums,0.01*Variable(self.xp.ones(mask_sums.shape).astype("f")))
        h = h/mask_sums + b
         
        mask_new = Variable(mask_new)
        h = F.where(mask_new_b, h, Variable(self.xp.zeros(h.shape).astype("f"))) 

        #elif self.sample=="up":
        #    h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
        #    h = self.c(h)
        #else:
        #    print("unknown sample method %s"%self.sample)
        if self.bn:
            h = self.batchnorm(h)
        if self.noise:
            h = add_noise(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h, mask_new

class PartialConvCompletion(chainer.Chain):
    """
    U-Net style network
    
    input     output
      l         l
    conv00 -> conv10
      l         l
    conv01 -> conv11
      l         l
    conv02 -> conv12
      i         i
      i         i
      i         i
    conv0n -> conv1n
         l    l
       conv0(n+1)

    output: h_dict['PConv10'] --- output of conv10

    Encode stage:
                 Input -> (PConv00) -> h_dict['PConv_00'] 64x128x128    
    h_dict['PConv_00'] -> (PConv01) -> h_dict['PConv_01'] 128x64x64                 
    h_dict['PConv_01'] -> (PConv02) -> h_dict['PConv_02'] 256x32x32                 
    h_dict['PConv_02'] -> (PConv03) -> h_dict['PConv_03'] 512x16x16                 
    h_dict['PConv_03'] -> (PConv04) -> h_dict['PConv_04'] 512x8x8                 
    h_dict['PConv_04'] -> (PConv05) -> h_dict['PConv_05'] 512x4x4
    h_dict['PConv_05'] -> (PConv06) -> h_dict['PConv_06'] 512x2x2

    Decode stage:
    
    dec: h_dict['PConv_06'] ->(up)--v   
    enc: h_dict['PConv_05'] ->(PConv_16)-> h_dict['PConv_16'] 512x4x4 
 
    dec: h_dict['PConv_16'] ->(up)--v   
    enc: h_dict['PConv_04'] ->(PConv_15)-> h_dict['PConv_15'] 512x8x8 
 
    dec: h_dict['PConv_15'] ->(up)--v 
    enc: h_dict['PConv_03'] ->(PConv_14)-> h_dict['PConv_14'] 512x16x16 

    dec: h_dict['PConv_14'] ->(up)--v   
    enc: h_dict['PConv_02'] ->(PConv_13)-> h_dict['PConv_13'] 256x32x32 
 
    dec: h_dict['PConv_13'] ->(up)--v   
    enc: h_dict['PConv_01'] ->(PConv_12)-> h_dict['PConv_12'] 128x64x64 
 
    dec: h_dict['PConv_12'] ->(up)--v   
    enc: h_dict['PConv_00'] ->(PConv_11)-> h_dict['PConv_11'] 64x128x128
 
    dec: h_dict['PConv_11'] ->(up)--v   
    enc:              Input ->(PConv_10)-> h_dict['PConv_10'] 3x256x256 
 
    """
    def __init__(self,ch0=3,input_size=256,layer_size=7): #input_size=512(2^9) in original paper but 256(2^8) in this implementation
        if 2**(layer_size+1) != input_size:
            raise AssertionError
        enc_layers = {}
        dec_layers = {}
        #encoder layers
        enc_layers['PConv_00'] = PConv(ch0, 64, bn=False, sample='down-7') #(1/2)^1
        enc_layers['PConv_01'] = PConv(64, 128, sample='down-5') #(1/2)^2
        enc_layers['PConv_02'] = PConv(128, 256, sample='down-5') #(1/2)^3
        enc_layers['PConv_03'] = PConv(256, 512, sample='down-3') #(1/2)^3
        for i in range(4,layer_size):     
            enc_layers['PConv_0'+str(i)] = PConv(512, 512, sample='down-3') #(1/2)^5
        
        #decoder layers
        for i in range(4,layer_size):
            dec_layers['PConv_1'+str(i)] = PConv(512*2, 512, activation=F.leaky_relu) 
        dec_layers['PConv_13'] = PConv(512+256, 256, activation=F.leaky_relu) 
        dec_layers['PConv_12'] = PConv(256+128, 128, activation=F.leaky_relu) 
        dec_layers['PConv_11'] = PConv(128+64, 64, activation=F.leaky_relu) 
        dec_layers['PConv_10'] = PConv(64+ch0, ch0, bn=False, activation=None)
        self.layer_size = layer_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        super(PartialConvCompletion, self).__init__(**enc_layers,**dec_layers)
        
    def __call__(self, x, x_mask):
        h_dict = {}
        mask_dict = {}
        
        #print("Encode stage")
        #print("[new step]: input -> PConv_00")
        #print("input shape:",x.shape)
        #print("mask shape:",x_mask.shape)
        h_dict['PConv_00'],mask_dict['PConv_00'] = self.enc_layers['PConv_00'](x,x_mask)
        key_prev = 'PConv_00'
        #print("PConv_00 sum: ",self.xp.sum(h_dict['PConv_00'].data))
        for i in range(1,self.layer_size):
            key = 'PConv_0'+str(i) 
            #print("[new step]: ",key_prev," -> ",key)
            #print("input shape:",h_dict[key_prev].shape)
            #print("mask shape:",mask_dict[key_prev].shape)
            h_dict[key], mask_dict[key] = self.enc_layers[key](h_dict[key_prev],mask_dict[key_prev])
            key_prev = key
            #print(key," sum: ",self.xp.sum(h_dict[key].data))
        
        #print("Decode stage") 
        #key_prev should be PConv06
        for i in reversed(range(self.layer_size-1)):
            enc_in_key = 'PConv_0'+str(i)
            dec_out_key = "PConv_1"+str(i+1)
            #print("[new step]:")
            #print("h_dict['",enc_in_key,"'] ---l")
            #print("h_dict['",key_prev,"'] --- h_dict['",dec_out_key,"']")
            #print("input enc shape:",h_dict[enc_in_key].shape)
            
            #unpooling (original paper used unsampling)
            h = F.unpooling_2d(h_dict[key_prev], 2, 2, 0, cover_all=False)
            mask = F.unpooling_2d(mask_dict[key_prev], 2, 2, 0, cover_all=False)
            #print("unpooled input dec shape:",h.shape)
            #print("unpooled input mask shape:",mask.shape)
            
            h = F.concat([h_dict[enc_in_key],h],axis=1) 
            mask = F.concat([mask_dict[enc_in_key],mask],axis=1) 
            h_dict[dec_out_key], mask_dict[dec_out_key] = self.dec_layers[dec_out_key](h,mask)
            key_prev = dec_out_key
            #print(dec_out_key," sum: ",self.xp.sum(h_dict[dec_out_key].data))
        #last step 
        dec_out_key = "PConv_10"
        #print("[new step]:")
        #print("                input ---l")
        #print("h_dict['",key_prev,"'] --- h_dict['PConv_10']")
        #print("input shape:",x.shape)
        
        #unpooling (original paper used unsampling)
        h = F.unpooling_2d(h_dict[key_prev], 2, 2, 0, cover_all=False)
        mask = F.unpooling_2d(mask_dict[key_prev], 2, 2, 0, cover_all=False)
        #print("unpooled input dec shape:",h.shape)
        #print("unpooled input mask shape:",mask.shape)
        
        h = F.concat([x,h],axis=1) 
        mask = F.concat([x_mask,mask],axis=1) 
        h_dict[dec_out_key], mask_dict[dec_out_key] = self.dec_layers[dec_out_key](h,mask)
        #print(dec_out_key," sum: ",self.xp.sum(h_dict[dec_out_key].data))

        return h_dict[dec_out_key] 


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, noise=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='none-9':
            layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
        elif sample=='none-7':
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
        elif sample=='none-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        else:
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            if self.noise:
                layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x, test):
        if self.sample=="down" or self.sample=="none" or self.sample=='none-9' or self.sample=='none-7' or self.sample=='none-5':
            h = self.c(x)
        elif self.sample=="up":
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.c(h)
        else:
            print("unknown sample method %s"%self.sample)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if self.noise:
            h = add_noise(h, test=test)
        if self.dropout:
            h = F.dropout(h, train=not test)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Generator_ResBlock_6(chainer.Chain):
    def __init__(self):
        super(Generator_ResBlock_6, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = CBR(128, 64, bn=True, sample='up'),
            c11 = CBR(64, 32, bn=True, sample='up'),
            c12 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, test=False, volatile=False):
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        return h

class Generator_ResBlock_9(chainer.Chain):
    def __init__(self):
        super(Generator_ResBlock_9, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = ResBlock(128, bn=True),
            c11 = ResBlock(128, bn=True),
            c12 = ResBlock(128, bn=True),
            c13 = CBR(128, 64, bn=True, sample='up'),
            c14 = CBR(64, 32, bn=True, sample='up'),
            c15 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, test=False, volatile=False):
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        h = self.c13(h, test=test)
        h = self.c14(h, test=test)
        h = self.c15(h, test=test)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_ch=3, n_down_layers=4):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        self.n_down_layers = n_down_layers

        layers['c0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
        base = 64

        for i in range(1, n_down_layers):
            layers['c'+str(i)] = CBR(base, base*2, bn=True, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
            base*=2

        layers['c'+str(n_down_layers)] = CBR(base, 1, bn=False, sample='none', activation=None, dropout=False, noise=True)

        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, test=False):
        h = self.c0(x_0, test=test)

        for i in range(1, self.n_down_layers+1):
            h = getattr(self, 'c'+str(i))(h, test=test)

        return h


class Completion(chainer.Chain):
    def __init__(self, in_ch=3):
        super(Completion,self).__init__(
            conv1 = L.Convolution2D(in_ch,   64,  ksize=5, stride=1, pad=2), 

            conv2 = L.Convolution2D(64, 128, ksize=3, stride=2, pad=1),
            conv3 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            conv4 = L.Convolution2D(128, 256, ksize=3, stride=2, pad=1), #became half
            conv5 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv6 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv7 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=2, dilate=2),
            conv8 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=4, dilate=4),
            conv9 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=8, dilate=8),
            conv10 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=16, dilate=16),
            conv11 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv12 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv13 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            conv14 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            conv15 = L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1),
            conv16 = L.Convolution2D(64, 32, ksize=3, stride=1, pad=1),
            conv17 = L.Convolution2D(32, 3, ksize=3, stride=1, pad=1))

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.relu(self.conv15(h))
        h = F.relu(self.conv16(h))
        y = F.sigmoid(self.conv17(h))
        return y

class Simplegen(chainer.Chain):
    def __init__(self):
        super(Simplegen, self).__init__(
            fc1 = L.Linear(None, 500),
            fc2 = L.Linear(None, 500),
            fc3 = L.Linear(None, 256*256*3))

    def __call__(self, x):
        B,ch,H,W = x.shape
        h = F.sigmoid(self.fc1(x))
        h = F.sigmoid(self.fc2(h))
        h = F.sigmoid(self.fc3(h))
        h = F.reshape(h,(B,ch,H,W))
        return h

class Local_Discriminator(chainer.Chain):
    def __init__(self, in_ch=3):
        super(Local_Discriminator,self).__init__(
            conv1 = L.Convolution2D(in_ch,   64,  ksize=4, stride=2, pad=1),
            conv2 = L.Convolution2D(64,   128,  ksize=4, stride=2, pad=1),
            conv3 = L.Convolution2D(128,   256,  ksize=4, stride=2, pad=1),
            conv4 = L.Convolution2D(256,   512,  ksize=4, stride=2, pad=1),
            conv5 = L.Convolution2D(512,   512,  ksize=4, stride=2, pad=1),
            fc = L.Linear(512 * 4 * 4, 1024))
    
    def __call__(self, x): #Variable 128x128
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        y = F.relu(self.fc(h))
        return y

class Global_Discriminator(chainer.Chain):
    def __init__(self, in_ch=3):
        super(Global_Discriminator,self).__init__(
            conv1 = L.Convolution2D(in_ch,   64,  ksize=4, stride=2, pad=1),
            conv2 = L.Convolution2D(64,   128,  ksize=4, stride=2, pad=1),
            conv3 = L.Convolution2D(128,   256,  ksize=4, stride=2, pad=1),
            conv4 = L.Convolution2D(256,   512,  ksize=4, stride=2, pad=1),
            conv5 = L.Convolution2D(512,   512,  ksize=4, stride=2, pad=1),
            conv6 = L.Convolution2D(512,   512,  ksize=4, stride=2, pad=1),
            fc = L.Linear(512 * 4 * 4, 1024))

    def __call__(self, x): #Variable 256x256
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        y = F.relu(self.fc(h))
        return y

class FC(chainer.Chain):
    def __init__(self):
        super(FC,self).__init__(
            fc = L.Linear(2048, 1))

    def __call__(self,local_out,global_out):
        h = F.concat([local_out,global_out],axis=-1)
        return F.sigmoid(self.fc(h))






