"""Classes related to different NN architectures"""

from .unet import UNet2D, UNet3D
from .psf_net import HybridUNet
from .regression_net import RegressionNet2D
from .layers import InterpUpSampling2D, InterpUpSampling3D
