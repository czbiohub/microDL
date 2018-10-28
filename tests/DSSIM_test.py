import numpy as np
from keras import backend as K
from micro_dl.train.losses import dssim_loss
import tensorflow as tf

#Shape should be (batch,x,y,channels)
imga = np.random.normal(size=(4,256,256,3))
imgb = np.random.normal(size=(4,256,256,3))

ssim1 = K.eval(tf.image.ssim(K.variable(imga), K.variable(imgb), max_val=1.0))
ssim2 = K.eval(tf.image.ssim(K.variable(imga), K.variable(imga), max_val=1.0))
resulting_loss1 = K.eval(dssim_loss(K.variable(imga),K.variable(imgb)))
resulting_loss2 = K.eval(dssim_loss(K.variable(imga),K.variable(imga)))

print ("Loss for different images: %.2f" % resulting_loss1)
print ("Loss for same image: %.2f" % resulting_loss2)