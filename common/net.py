import numpy as np
import sys, os

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

class PConv(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='none-3', activation=F.relu, dropout=False, noise=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        layers = {}
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

