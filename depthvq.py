from tinygrad import Device
from tinygrad.nn.state import load_state_dict, torch_load
from vqvae import Encoder, Decoder, transpose_and_clip
from dataclasses import dataclass

@dataclass
class CompressorConfig:
   in_channels:  int = 1
   out_channels: int = 1
   ch_mult: tuple[int,...] = (1,1,2,2,4)
   attn_resolutions: tuple[int] = (16,)
   num_res_blocks: int = 2
   resolution: int = 256
   z_channels: int = 128
   vocab_size: int = 256
   ch: int = 128
   dropout: float = 0.2

   @property
   def num_resolutions(self):
      return len(self.ch_mult)

   @property
   def quantized_resolution(self):
      return self.resolution // 2**(self.num_resolutions-1)

class DepthVQ:
   mappings = {
      "quant_conv": [("weight",0), ("bias",0)],
      "post_quant_conv": [("weight",1), ("bias",None)],
      "quantize": { "_embedding": [("weight",1)] },
   }

   def __init__(self, config=CompressorConfig()):
      self.encoder = Encoder(config)
      self.decoder = Decoder(config)

   def __map_state_dict(self, state_dict, mapping, prefix):
      if isinstance(mapping, list):
         for item, dim in mapping:
            key = f"{prefix}.{item}"
            if dim is None:
               enc_w = dec_w = state_dict[key]
            else:
               enc_w, dec_w = state_dict[key].to(Device.DEFAULT).chunk(2, dim=dim)
            state_dict[f"encoder.{key}"] = enc_w
            state_dict[f"decoder.{key}"] = dec_w
         return state_dict
      assert isinstance(mapping, dict)
      for key, item in mapping.items():
         self.__map_state_dict(state_dict, item, f"{prefix}.{key}" if prefix else key)

   def load_from_pretrained(self) -> 'DepthVQ':
      state_dict = torch_load("./weights/model.ckpt")["state_dict"]
      state_dict["quantize._embedding.weight"] = state_dict["quantize.embedding.weight"]
      self.__map_state_dict(state_dict, self.mappings, "")
      load_state_dict(self, state_dict) # type: ignore
      return self

if __name__ == "__main__":
   from tinygrad import Tensor, dtypes
   from PIL import Image
   import numpy as np
   
   model = DepthVQ().load_from_pretrained()
   img = Tensor(np.array(Image.open("./img_depth.png")))
   x = img.rearrange('h w -> 1 1 h w').cast(dtypes.float32)

   tokens = model.encoder(x).argmax(axis=-1).realize()
   print(tokens.shape)
   x = transpose_and_clip(model.decoder(tokens).realize()).squeeze()
   print(x.shape)

   Image.fromarray(x.numpy()).show()
