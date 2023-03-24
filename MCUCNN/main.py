import torchsummary
import torch
from torchsummary.summarizer import SMA

from MobileNetV2 import MobileNetV2
from Proxyless import Proxyless
from MnasNet import MnasNet
from SmallCifar import SmallCifar

def calibrate_func(model, shape):
    input1 = torch.rand(shape)
    output = model(input1)
    
# SmallCifar
model = SmallCifar()
model.eval()
shape=(1, 3, 32, 32)
sm = SMA(model, shape)
sm.init()
sm.prepare()
sm.calibrate(calibrate_func, shape)
sm.concise("SmallCifar")

# MobileNetV2
model = MobileNetV2()
model.eval()
shape=(1, 3, 144, 144)
sm = SMA(model, shape)
sm.prepare()
sm.calibrate(calibrate_func, shape)
sm.concise("MobileNetV2")

# Proxyless
model = Proxyless()
model.eval()
shape=(1, 3, 176, 176)
sm = SMA(model, shape)
sm.prepare()
sm.calibrate(calibrate_func, shape)
sm.concise("Proxyless")

# MnasNet

model = MnasNet()
model.eval()
shape=(1, 3, 96, 96)
sm = SMA(model, shape)
sm.prepare()
sm.calibrate(calibrate_func, shape)
sm.concise("MnasNet")