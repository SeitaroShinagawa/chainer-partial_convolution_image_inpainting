import os
import copy

import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from utils import batch_postprocess_images
from PIL import Image

def evaluation(model, test_image_folder, image_size=256):
    @chainer.training.make_extension()
    def _eval(trainer, it):
        xp = model.xp
        batch = it.next()
        batchsize = len(batch)

        x = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        m = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            m[i, :] = xp.asarray(batch[i][1])
        mask_b = xp.array(m.astype("bool"))

        I_gt = Variable(x)
        M = Variable(m)
        M_b = Variable(mask_b)
        
        I_out = model(x, m)
        I_comp = F.where(M_b,I_gt,I_out)

        img = I_comp.data.get()

        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Icomp.jpg")

        img = I_out.data.get()

        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_Iout.jpg")

        img = M.data.get()

        img = batch_postprocess_images(img, int(batchsize/2), 2)
        Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_mask.jpg")

    def evaluation(trainer):
        it = trainer.updater.get_iterator('test')
        _eval(trainer, it)

    return evaluation
