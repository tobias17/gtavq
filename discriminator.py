"""
Adapted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
"""
from tinygrad import Tensor, nn
from functools import partial

class NLayerDiscriminator:
   def __init__(self, input_nc:int=3, ndf:int=64, n_layers:int=3):
      kw = 4
      padw = 1
      self.main = [
         nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
         partial(Tensor.leakyrelu, neg_slope=0.2),
      ]
      nf_mult = 1
      nf_mult_prev = 1
      for n in range(1, n_layers+1):
         nf_mult_prev = nf_mult
         nf_mult = min(2 ** n, 8)
         self.main += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            partial(Tensor.leakyrelu, neg_slope=0.2),
         ]

      self.main.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential(self.main) # type: ignore

def hinge_d_loss(logits_real:Tensor, logits_fake:Tensor) -> Tensor:
   loss_real = (1.0 - logits_real).relu().mean()
   loss_fake = (1.0 + logits_fake).relu().mean()
   return 0.5 * (loss_real + loss_fake)

if __name__ == "__main__":
   gan = NLayerDiscriminator()
   x = Tensor.randn(8, 3, 256, 128).realize()
   print(gan(x).realize().shape)
