from zennit.core import collect_leaves,stabilize
import zennit.canonizers as zcanon
from zennit.layer import Sum
from zennit.image import imgify
import torch
import torch.nn.functional as F
import torch.nn as nn
algc = False

class InterpolateWrapper(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=True):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, 
            size=self.size, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

class Cat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)

class SequentialMergeBatchNorm(zcanon.SequentialMergeBatchNorm):
    # Zennit does not set bn.epsilon to 0, resulting in an incorrect canonization. We solve this issue.
    def __init__(self):
        super(SequentialMergeBatchNorm, self).__init__()

    def apply(self, root_module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.linear_type) and isinstance(leaf, self.batch_norm_type):
                if last_leaf.weight.shape[0] == leaf.weight.shape[0]:
                    instance = self.copy()
                    instance.register((last_leaf,), leaf)
                    instances.append(instance)
            last_leaf = leaf

        return instances

    def merge_batch_norm(self, modules, batch_norm):
        self.batch_norm_eps = batch_norm.eps
        super(SequentialMergeBatchNorm, self).merge_batch_norm(modules, batch_norm)
        batch_norm.eps = 0.

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super(SequentialMergeBatchNorm, self).remove()
        self.batch_norm.eps = self.batch_norm_eps

# Canonizer for PIDNet
class PIDNetCanonizer(zcanon.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        # BasicBlock
        if module.__class__.__name__ == "BasicBlock":
            return {
                'forward': cls.forward_basicblock.__get__(module),
                'canonizer_sum': Sum()
            }

        # Bottleneck
        if module.__class__.__name__ == "Bottleneck":
            return {
                'forward': cls.forward_bottleneck.__get__(module),
                'canonizer_sum': Sum()
            }

        # PAPPM
        if module.__class__.__name__ == "PAPPM":
            return {
                'forward': cls.forward_pappm.__get__(module),
                'canonizer_sum': Sum()
            }

        # Light_Bag
        if module.__class__.__name__ == "Light_Bag":
            return {
                'forward': cls.forward_lightbag_branch_i_only.__get__(module),
                'canonizer_sum': Sum()
            }

        return None

    @staticmethod
    def forward_basicblock(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.canonizer_sum(torch.stack([out, residual], dim=-1))

        if self.no_relu:
            return out
        else:
            return self.relu(out)


    @staticmethod
    def forward_bottleneck(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.canonizer_sum(torch.stack([out, residual], dim=-1))
        if self.no_relu:
            return out
        else:
            return self.relu(out)

    @staticmethod
    def forward_pappm(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []
        self.interp1 = InterpolateWrapper(size=[height, width], mode='bilinear', align_corners=algc)
        self.interp2 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)
        self.interp3 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)
        self.interp4 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)
        self.cat = Cat(dim=1)

        x_ = self.scale0(x)

        s1 = self.interp1(self.scale1(x))
        s2 = self.interp1(self.scale2(x))
        s3 = self.interp1(self.scale3(x))
        s4 = self.interp1(self.scale4(x))
        
        scale_list.append(self.canonizer_sum(torch.stack([s1, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s2, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s3, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s4, x_], dim=-1)))
        # scale_list.append(self.canonizer_sum(torch.stack([s2, x_], dim=-1)))

        
        scale_out = self.scale_process(self.cat(*scale_list))
       
        # Here is some error with gradient, that dimensions do not correspond (on forward pass no problem). 
        out = self.compression(self.cat(x_,scale_out)) + self.shortcut(x)
        return out
    
    @staticmethod
    def forward_lightbag_branch_i_only(self, p, i, d):
        # Detaching branches P and D here 
        edge_att = torch.sigmoid(d).detach()
        
        p_add = self.conv_p((1-edge_att)*i + p.detach())
        i_add = self.conv_i(i + edge_att*p.detach())
        
        return self.canonizer_sum(torch.stack([p_add, i_add], dim=-1))
    

# Top-level composite canonizer to combine canonization strategies
class CanonizerPIDNet(zcanon.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            PIDNetCanonizer(),
        ))