#!/usr/bin/env python
import argparse
import os
import chainer
from chainer import training
from chainer import cuda, serializers
from chainer.training import extension
from chainer.training import extensions
import sys
import common.net as net
import datasets
from updater import *
from evaluation import *
from chainer.links import VGG16Layers
import common.paths as paths

def main():
    parser = argparse.ArgumentParser(
        description='Train Completion Network')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--eval_folder', '-e', default='generated_results',
                        help='Directory to output the evaluation result')

    parser.add_argument("--load_model", help='completion model path')

    parser.add_argument("--resize_to", type=int, default=256, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to')
    parser.add_argument("--load_dataset", default='place2_test', help='load dataset')
    #parser.add_argument("--layer_n", type=int, default=7, help='number of layers')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
    
    #load completion model
    model = getattr(net, "PartialConvCompletion")(ch0=3,input_size=args.crop_to)

    #load vgg_model
    print("loading vgg16 ...")
    vgg = VGG16Layers()
    print("ok")

    if args.load_model != '':
        serializers.load_npz(args.load_model, model)
        print("Completion model loaded")

    if not os.path.exists(args.eval_folder):
         os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0:
        model.to_gpu()
        vgg.to_gpu()
        print("use gpu {}".format(args.gpu))

    val_dataset = getattr(datasets, args.load_dataset)(paths.val_place2, mask_path="mask/256", resize_to=args.resize_to, crop_to=args.crop_to)
    val_iter = chainer.iterators.SerialIterator(
        val_dataset, args.batch_size)

    #test_dataset = horse2zebra_Dataset_train(flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)

    #test_iter = chainer.iterators.SerialIterator(train_dataset, 8)


    #generate results
    xp = model.xp
    batch = val_iter.next()
    batchsize = len(batch)

    image_size = args.crop_to
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

    img = x.get()

    img = batch_postprocess_images(img, batchsize, 1)
    Image.fromarray(img).save(args.eval_folder+"/generated_3_Igt.jpg")

    img = I_comp.data.get()

    img = batch_postprocess_images(img, batchsize, 1)
    Image.fromarray(img).save(args.eval_folder+"/generated_2_Icomp.jpg")

    img = I_out.data.get()

    img = batch_postprocess_images(img, batchsize, 1)
    Image.fromarray(img).save(args.eval_folder+"/generated_1_Iout.jpg")

    img = M.data.get()

    img = batch_postprocess_images(img, batchsize, 1)
    Image.fromarray(img).save(args.eval_folder+"/generated_0_mask.jpg")

if __name__ == '__main__':
    main()
