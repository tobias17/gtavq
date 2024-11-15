import os, sys
sys.path.append("../taming-transformers")
import main
from taming.models.vqgan import VQModel
import numpy as np
import torch, torchvision
from PIL import Image

def load_taming_vqgan():
   ddconfig = {
      "double_z": False,
      "z_channels": 128,
      "resolution": 256,
      "in_channels": 1,
      "out_ch": 1,
      "ch": 128,
      "ch_mult": [ 1,1,2,2,4],
      "num_res_blocks": 2,
      "attn_resolutions": [16],
      "dropout": 0.0,
   }
   lossconfig = {
      "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
      "params": {
         "disc_conditional": False,
         "disc_in_channels": 1,
         "disc_start": 10000,
         "disc_weight": 0.8,
         "codebook_weight": 1.0,
      }
   }
   return VQModel(ddconfig, lossconfig, n_embed=256, embed_dim=256, ckpt_path="./weights/model.ckpt")

def to_input(im:Image.Image) -> torch.Tensor:
   x = torch.Tensor(np.array(im)).cuda().float()
   x = x.reshape(1, *x.shape, 1)
   x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
   return (x / 127.0) - 1.0

def from_output(x:torch.Tensor) -> Image.Image:
   x = x[0]
   x = torch.clamp(x, -1.0, 1.0)
   x = (x+1.0) / 2.0
   x = x.transpose(0,1).transpose(1,2).squeeze(-1)
   x = (x.cpu().numpy() * 255).astype(np.uint8)
   return Image.fromarray(x)

if __name__ == "__main__":
   model = load_taming_vqgan().cuda().eval()

   x_in = to_input(Image.open("./img_depth.png"))
   with torch.no_grad():
      quant, _, _ = model.encode(x_in)
      print(quant.shape)
      x_out = model.decode(quant)
   from_output(x_out).show()
