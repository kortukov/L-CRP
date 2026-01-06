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

    def canonize(self, layer, additional_canonizers=None, submodule_names=None):
        h1 = PIDNetBaseCanonizer().apply(layer)
        self.handles += h1
        if not isinstance(additional_canonizers, list):
            if not isinstance(submodule_names, list):
                additional_canonizers = [additional_canonizers]
                submodule_names = [submodule_names]
            else:
                additional_canonizers=[additional_canonizers]*len(submodule_names)
        for i, canonizer in enumerate(additional_canonizers):
            obj = layer
            if submodule_names[i] is not None:
                obj = getattr(obj, submodule_names[i])
            if canonizer is not None:
                h2 = canonizer.apply(obj)
                self.handles += h2

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
        for k in i_branch + p_branch + d_branch:
            self.canonize(getattr(model, k), CorrectSequentialMergeBatchNorm())
        TReLU_modules = [
            "scale1",
            "scale2",
            "scale3",
            "scale4",
            "scale0",
            "scale_process",
            "compression",
            "shortcut",
        ]
        self.canonize(model.spp, ThreshReLUMergeBatchNorm(), TReLU_modules)
        # self.ConvBNCanonization(model.dfm) # TODO
        # Prediction Head
        segheads = ["final_layer"] + ["seghead_p", "seghead_d"] if model.augment else []
        for sh_layer in segheads:
            self.canonize(
                getattr(model, sh_layer),
                [CorrectSequentialMergeBatchNorm(), ThreshReLUMergeBatchNorm()],
                ["sequential", "sequential"],
            )

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
