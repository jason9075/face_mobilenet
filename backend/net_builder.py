from enum import Enum

from backend.arch.mobilefacenet import MobileFaceNet
from backend.arch.mobilenet_v1 import MobileNetV1
from backend.arch.mobilenet_v2 import MobileNetV2
from backend.arch.squeezenet import SqueezeNet
from backend.arch.resnet34 import ResNet34
from backend.arch.resnet50 import ResNet50
from backend.final_layers import *


class Arch(Enum):
    MOBILE_NET_V1 = MobileNetV1
    MOBILE_NET_V2 = MobileNetV2
    MOBILE_FACE_NET = MobileFaceNet
    SQUEEZE_NET = SqueezeNet
    RES_NET34 = ResNet34
    RES_NET50 = ResNet50


class FinalLayer(Enum):
    GDC = gdc
    G = g_type


class NetBuilder:
    def __init__(self):
        self.input_node = None
        self.is_train_node = None
        self.arch = None
        self.final_layer = None

    def input_and_train_node(self, input_node, is_train_node):
        self.input_node = input_node
        self.is_train_node = is_train_node

        return self

    def arch_type(self, arch):
        if arch not in Arch:
            raise KeyError('arch \'%s\' is not Enum!' % arch)
        self.arch = arch.value

        return self

    def final_layer_type(self, final_layer):
        self.final_layer = final_layer

        return self

    def build(self):
        net = self.arch(self.input_node, self.is_train_node)
        net = self.final_layer(net)
        # net = tf.nn.relu6(net)
        return net
