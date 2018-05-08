import os
import copy

import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F

import numpy as np
import os
import cv2

from PIL import Image

def evaluation(model, test_image_folder, image_size=256, side=2):
    @chainer.training.make_extension()
    def _eval(trainer, it):
        xp = gen_g.xp
        batch = it.next()
        batchsize = len(batch)

        #x = []
        x = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        m = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            m[i, :] = xp.asarray(batch[i][1])

        I_gt = Variable(x)
        M = Variable(m)
        M_b = Variable(m.astype("bool"))
        
        I_out = model(x, mask, test=True)
        I_comp = F.where(M_b,I_gt,I_out)

        img = I_comp.data.get()

        img_c = img.reshape((side, side, 3, image_size, image_size))
        img_c = img_c.transpose(0,1,3,4,2)
        img_c = (img + 1) *127.5
        img_c = np.clip(img_c, 0, 255)
        img_c = img_c.astype(np.uint8)
        img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))
        Image.fromarray(img_c).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Icomp.jpg")

        img = I_out.data.get()

        img_c = img.reshape((side, side, 3, image_size, image_size))
        img_c = img_c.transpose(0,1,3,4,2)
        img_c = (img + 1) *127.5
        img_c = np.clip(img_c, 0, 255)
        img_c = img_c.astype(np.uint8)
        #img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))[:,:,::-1]
        img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))
        Image.fromarray(img_c).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Iout.jpg")

        img = M.data.get()

        img_c = img.reshape((side, side, 1, image_size, image_size))
        img_c = img_c.transpose(0,1,3,4,2)
        img_c = img*255
        img_c = np.clip(img_c, 0, 255)
        img_c = img_c.astype(np.uint8)
        img_c = img_c.reshape((side, side, image_size, image_size, 1)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 1))
        Image.fromarray(img_c).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_mask.jpg")

    def evaluation(trainer):
        it = trainer.updater.get_iterator('test')
        _eval(trainer, it)

    return evaluation
