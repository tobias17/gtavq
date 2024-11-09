"""
Adapted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
"""
from tinygrad import Tensor, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, get_parameters, torch_load
from vgg16 import vgg16
from typing import List, Dict

class ScalingLayer:
   def __init__(self):
      self.shift = Tensor([-.030, -.088, -.188]).reshape(1, -1, 1, 1)
      self.scale = Tensor([ .458,  .448,  .450]).reshape(1, -1, 1, 1)
   def __call__(self, x:Tensor) -> Tensor:
      return (x - self.shift) / self.scale

class FrozenVgg16:
   def __init__(self):
      vgg_features = vgg16().load_from_pretrained().features
      self.slice1 = vgg_features[  : 4]
      self.slice2 = vgg_features[ 4: 9]
      self.slice3 = vgg_features[ 9:16]
      self.slice4 = vgg_features[16:23]
      self.slice5 = vgg_features[23:30]
      for w in get_parameters(self):
         w.requires_grad = False
   
   def __call__(self, x:Tensor) -> List[Tensor]:
      actv = []
      for slice_n in (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5):
         x = x.sequential(slice_n)
         actv.append(x)
      return actv

class NetLinLayer:
   def __init__(self, chn_in:int, chn_out:int=1):
      self.model = [
         Tensor.dropout,
         nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
      ]

def normalize_tensor(x:Tensor, eps:float=1e-10) -> Tensor:
   norm_factor = Tensor.sqrt(Tensor.sum(x**2, axis=1, keepdim=True))
   return x / (norm_factor + eps)

def spatial_average(x:Tensor, keepdim=True) -> Tensor:
   return x.mean([2,3], keepdim=keepdim)

class LPIPS:
   def __init__(self):
      self.scaling_layer = ScalingLayer()
      self.net  = FrozenVgg16()
      self.chns = [64, 128, 256, 512, 512]
      self.lin0 = NetLinLayer(self.chns[0])
      self.lin1 = NetLinLayer(self.chns[1])
      self.lin2 = NetLinLayer(self.chns[2])
      self.lin3 = NetLinLayer(self.chns[3])
      self.lin4 = NetLinLayer(self.chns[4])

   def load_from_pretrained(self, url:str="https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1", name:str="vgg_lpips.pth") -> 'LPIPS':
      mdl = { f"lin{i}": getattr(self, f"lin{i}") for i in range(5) }
      load_state_dict(mdl, torch_load(str(fetch(url, name))), strict=True)
      for w in get_parameters(self):
         w.requires_grad = False
      return self

   def __call__(self, input:Tensor, target:Tensor) -> Tensor:
      in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
      outs0, outs1 = self.net(in0_input), self.net(in1_input)
      feats0, feats1, diffs = {}, {}, {}
      lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
      for kk in range(len(self.chns)):
         feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
         diffs[kk] = Tensor.square(feats0[kk] - feats1[kk])
      
      res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
      val = res[0]
      for l in range(1, len(self.chns)):
         val += res[l]
      return val

if __name__ == "__main__":
   LPIPS().load_from_pretrained()
