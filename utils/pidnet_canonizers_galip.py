from LCRP.models.pidnet import PIDNet
from LCRP.utils.base_canonizers import (
    CorrectSequentialMergeBatchNorm,
    ThreshReLUMergeBatchNorm,
    SequentialMergeBatchNormtoRight,
)
from LCRP.models.model_utils import BasicBlock, Bottleneck, PagFM, PAPPM
from torch.nn import Sequential, Module, AvgPool2d, BatchNorm2d
import torch
from torch.nn.functional import relu as ReLU
from copy import deepcopy
from zennit.types import ConvolutionTranspose
from zennit.core import collect_leaves
from LCRP.utils.pidnet_canonizers import PIDNetBaseCanonizer
from zennit.canonizers import AttributeCanonizer
import torchvision
from zennit.canonizers import Canonizer


class PIDNetCanonizer(Canonizer):
    # New Module that computes the same functions as model, including layer wrappers and canonizations

    def canonize(self, layer, additional_canonizer=None):
        h1 = PIDNetBaseCanonizer().apply(layer)
        self.handles+=h1
        if additional_canonizer is not None:
            h2 = additional_canonizer.apply(layer)
            self.handles += h2

    def BNConvCanonization(self, layer):
        h1 = PIDNetBaseCanonizer().apply(layer)
        h2 = CorrectSequentialMergeBatchNorm().apply(layer)
        self.handles += h1 + h2

    def BNReLUConvCanonization(self, layer):
        h1 = PIDNetBaseCanonizer().apply(layer)
        h2 = ThreshReLUMergeBatchNorm().apply(layer)
        self.handles += h1 + h2

    def register(self, model):
        self.handles = []
        self.threshbn = ThreshReLUMergeBatchNorm()
        # I Branch
        i_branch = ["conv1", "layer1", "layer2", "layer3", "layer4", "layer5"]

        # P Branch
        p_branch = ["compression3", "compression4", "layer3_", "layer4_", "layer5_"]

        # self.pag3 = PagFM(planes * 2, planes) TODO
        # self.pag4 = PagFM(planes * 2, planes)

        # D Branch
        d_branch = ["layer3_d", "layer4_d", "diff3", "diff4", "layer5_d"]
        for k in i_branch+p_branch+d_branch:
            self.canonize(getattr(model, k), CorrectSequentialMergeBatchNorm())
        self.canonize(model.spp, ThreshReLUMergeBatchNorm())
        # self.BNConvReLUCanonization(model.dfm)

        # Prediction Head
        # if model.augment:
        # self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
        # self.seghead_d = segmenthead(planes * 2, planes, 1)

        # self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

    def remove(self):
        self.handles.reverse()
        for h in self.handles:
            h.remove()

    def apply(self, module):
        if isinstance(module, PIDNet):
            instance = self.copy()
            instance.register(module)
            return [instance]
        else:
            return []
