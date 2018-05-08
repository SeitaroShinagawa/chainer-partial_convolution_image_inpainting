import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import six
import os

from chainer import cuda, optimizers, serializers, Variable
from chainer import training

from PIL import Image
import chainer.computational_graph as c

def calc_loss_perceptual(hout_dict,hcomp_dict,hgt_dict):
    layers = list(hout_dict.keys())
    layer_name =  layers[0]
    loss = F.mean_absolute_error(hout_dict[layer_name],hgt_dict[layer_name])
    loss += F.mean_absolute_error(hout_dict[layer_name],hgt_dict[layer_name])
    for layer_name in layers[1:]: 
        loss += F.mean_absolute_error(hcomp_dict[layer_name],hgt_dict[layer_name])
        loss += F.mean_absolute_error(hcomp_dict[layer_name],hgt_dict[layer_name])
    return loss

def vgg_extract(vgg_model, Img, layers=['pool1','pool2','pool3'],in_size=224):
    B,C,H,W = Img.shape #BGR [0,1] range
    Img = (Img + 1)*127.5
    Img_chanells = [F.expand_dims(Img[:,i,:,:],axis=1) for i in range(3)]
    Img_chanells[0] -= 103.939          #subtracted by [103.939, 116.779, 123.68] 
    Img_chanells[1] -= 116.779          #subtracted by [103.939, 116.779, 123.68] 
    Img_chanells[2] -= 123.68          #subtracted by [103.939, 116.779, 123.68] 
    Img = F.concat(Img_chanells,axis=1)
    limx = H - in_size
    limy = W - in_size
    xs = np.random.randint(0,limx,B)
    ys = np.random.randint(0,limy,B)
    lis = [F.expand_dims(Img[i,:,x:x+in_size,y:y+in_size],axis=0) for i,(x,y) in enumerate(zip(xs,ys))]
    Img_cropped = F.concat(lis,axis=0)
    return vgg_model(Img_cropped,layers=layers)

def calc_loss_style(hout_dict,hcomp_dict,hgt_dict):
    layers = hgt_dict.keys()
    for i,layer_name in enumerate(layers):
        B,C,H,W = hout_dict[layer_name].shape
        hout = F.reshape(hout_dict[layer_name],(B,C,H*W))
        hcomp = F.reshape(hcomp_dict[layer_name],(B,C,H*W))
        hgt = F.reshape(hgt_dict[layer_name],(B,C,H*W))
        
        hout_gram = F.batch_matmul(hout,hout,transb=True)
        hcomp_gram = F.batch_matmul(hcomp,hcomp,transb=True)
        hgt_gram = F.batch_matmul(hgt,hgt,transb=True)
        
        if i==0: 
            L_style_out = F.mean_absolute_error(hout_gram,hgt_gram)/(C*H*W)
            L_style_comp = F.mean_absolute_error(hcomp_gram,hgt_gram)/(C*H*W)
        else:
            L_style_out += F.mean_absolute_error(hout_gram,hgt_gram)/(C*H*W)
            L_style_comp += F.mean_absolute_error(hcomp_gram,hgt_gram)/(C*H*W)        

    return L_style_out + L_style_comp

def calc_loss_tv(Icomp, mask, xp=np):
    canvas = mask.data
    canvas[:,:,:,:-1] += mask.data[:,:,:,1:] #mask left overlap
    canvas[:,:,:,1:] += mask.data[:,:,:,:-1] #mask right overlap
    canvas[:,:,:-1,:] += mask.data[:,:,1:,:] #mask up overlap
    canvas[:,:,1:,:] += mask.data[:,:,:-1,:] #mask bottom overlap
    
    P = Variable(xp.sign(canvas-0.5)*0.5+1.0) #P region (hole mask: 1 pixel dilated region from hole)
    return F.mean_absolute_error(P[:,:,:,1:]*Icomp[:,:,:,1:],P[:,:,:,:-1]*Icomp[:,:,:,:-1])+ F.mean_absolute_error(P[:,:,1:,:]*Icomp[:,:,1:,:],P[:,:,:-1,:]*Icomp[:,:,:-1,:]) 


def imgcrop_batch(img,pos_list,size=128):
    B,ch,H,W = img.shape
    lis = [F.expand_dims(img[i,:,x:x+size,y:y+size],axis=0) for i,(x,y) in enumerate(pos_list)]
    return F.concat(lis,axis=0)

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.vgg, self.model = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._lambda3 = params['lambda3']
        self._lambda4 = params['lambda4']
        self._image_size = params['image_size']
        self._eval_foler = params['eval_folder']
        self._dataset = params['dataset']
        self._iter = 0
        xp = self.model.xp
        
        super(Updater, self).__init__(*args, **kwargs)

    """
    def save_images(self,img, w=2, h=3):
        img = cuda.to_cpu(img)
        img = img.reshape((w, h, 3, self._image_size, self._image_size))
        img = img.transpose(0,1,3,4,2)
        img = (img + 1) *127.5
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = img.reshape((w, h, self._image_size, self._image_size, 3)).transpose(0,2,1,3,4).reshape((w*self._image_size, h*self._image_size, 3))[:,:,::-1]
        Image.fromarray(img).save(self._eval_foler+"/iter_"+str(self._iter)+".jpg")
    """

    def update_core(self):
        xp = self.model.xp
        self._iter += 1
        batch = self.get_iterator('main').next() #img_processed (B,4,H,W), origin (B,3,H,W), mask (B,1,H,W)
        batchsize = len(batch)

        w_in = self._image_size

        zero_f = Variable(xp.zeros((batchsize, 3, w_in, w_in)).astype("f"))
        
        x_train = np.zeros((batchsize, 3, w_in, w_in)).astype("f")
        mask_train = np.zeros((batchsize, 3, w_in, w_in)).astype("f")
         
        for i in range(batchsize):
            x_train[i, :] = batch[i][0] #original image
            mask_train[i, :] = batch[i][1] #0-1 mask of c 
        
        x_train = xp.array(x_train)
        mask_train = xp.array(mask_train)
        mask_b = xp.array(mask_train.astype("bool"))
        
        I_gt = Variable(x_train)
        M = Variable(mask_train)
        M_b = Variable(mask_b)

        I_out = self.model(I_gt,M)
        I_comp = F.where(M_b,I_gt,I_out) #if an element of Mc_b is True, return the corresponded element of I_gt, otherwise return that of I_out)

        fs_I_gt = vgg_extract(self.vgg,I_gt) #feature dict
        fs_I_out = vgg_extract(self.vgg,I_out) #feature dict
        fs_I_comp = vgg_extract(self.vgg,I_comp) #feature dict

        opt_model = self.get_optimizer('model')

        #if self._learning_rate_anneal > 0 and self._iter % self._learning_rate_anneal_interval == 0:
        #    if opt_model.alpha > self._learning_rate_anneal:
        #        opt_model.alpha -= self._learning_rate_anneal

        L_valid = F.mean_absolute_error(M*I_out,M*I_gt)
        L_hole = F.mean_absolute_error((1-M)*I_out,(1-M)*I_gt) 
        
        L_perceptual = calc_loss_perceptual(fs_I_gt,fs_I_out,fs_I_comp)
        
        L_style = calc_loss_style(fs_I_out,fs_I_comp,fs_I_gt) #Loss style out and comp 
        L_tv = calc_loss_tv(I_comp, M, xp=xp)

        L_total = L_valid + self._lambda1 * L_hole + self._lambda2 * L_perceptual + \
                  self._lambda3 * L_style + self._lambda4 * L_tv
        
        #g = c.build_computational_graph([L_total])
        #with open("graph.dot","w") as o:
        #    o.write(g.dump())
        
        self.vgg.cleargrads()
        self.model.cleargrads()
        L_total.backward()
        opt_model.update()

        chainer.report({'L_valid': L_valid})
        chainer.report({'L_hole': L_hole})
        chainer.report({'L_perceptual': L_perceptual})
        chainer.report({'L_style': L_style})
        chainer.report({'L_tv': L_tv})

        #if self._iter%100 ==0:
        #    #img = xp.zeros((6, 3, w_in, w_in)).astype("f")
        #    img = xp.zeros((2, 3, w_in, w_in)).astype("f")
        #    img[0, : ] = I_comp.data[0]
        #    img[1, : ] = I_gt.data[0]
        #    #img[2, : ] = I_comp.data[1]
        #    #img[3, : ] = I_gt.data[1]
        #    #img[4, : ] = I_comp.data[2]
        #    #img[5, : ] = I.gt.data[2]
        #    img = cuda.to_cpu(img)
        #    #img = self._dataset.batch_postprocess_images(img, 3, 2)
        #    img = self._dataset.batch_postprocess_images(img, 1, 2)
        #    Image.fromarray(img).save(self._eval_foler+"/iter_"+str(self._iter)+".jpg")
