import numpy as np

def batch_postprocess_images(img, batch_w, batch_h):
    b, ch, w, h = img.shape
    img = img.reshape((batch_w, batch_h, ch, w, h))
    img = img.transpose(0,1,3,4,2)
    img = (img + 1) *127.5
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img.reshape((batch_w, batch_h, w, h, ch)).transpose(0,2,1,3,4).reshape((w*batch_w, h*batch_h, ch))
    return img

