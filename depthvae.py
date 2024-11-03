from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict
from dataclasses import dataclass
from vqvae import Encoder, Decoder
from PIL import Image
from queue import Queue
from threading import Thread, Event
import numpy as np
import os, time

@dataclass
class CompressorConfig:
   in_channels:  int = 1
   out_channels: int = 1
   ch_mult: tuple[int,...] = (1,1,2,2,4)
   attn_resolutions: tuple[int] = (16,)
   resolution: int = 256
   num_res_blocks: int = 2
   z_channels: int = 256
   vocab_size: int = 1024
   ch: int = 128
   dropout: float = 0.0

   @property
   def num_resolutions(self):
      return len(self.ch_mult)

   @property
   def quantized_resolution(self):
      return self.resolution // 2**(self.num_resolutions-1)

# @dataclass
# class CompressorConfig:
#    in_channels: int = 1
#    out_channels: int = 1
#    ch_mult: tuple[int,...] = (1,1,1,2,2,4)
#    attn_resolutions: tuple[int] = (8,)
#    resolution: int = 256
#    num_res_blocks: int = 2
#    z_channels: int = 256
#    vocab_size: int = 512
#    ch: int = 32
#    dropout: float = 0.1

#    @property
#    def num_resolutions(self):
#       return len(self.ch_mult)

#    @property
#    def quantized_resolution(self):
#       return self.resolution // 2**(self.num_resolutions-1)

class VQModel:
   def __init__(self, config=CompressorConfig()):
      self.enc = Encoder(config)
      self.dec = Decoder(config)
      self.dec.quantize = self.enc.quantize

def get_next_pack_path():
   ROOT = "./depthpacks"
   for splitdir in os.listdir(ROOT):
      for filename in os.listdir(f"{ROOT}/{splitdir}"):
         yield f"{ROOT}/{splitdir}/{filename}"

def train():
   Tensor.training = True
   Tensor.manual_seed(42)

   model = VQModel()

   TRAIN_DTYPE = dtypes.float32
   GLOBAL_BS = 16

   LEARNING_RATE = 2**-15
   optim = AdamW(get_parameters(model), lr=LEARNING_RATE)
   step_i = 0

   depthpack_getter = get_next_pack_path()

   @TinyJit
   def train_step(init_x:Tensor) -> Tensor:
      token_probs = model.enc(init_x)
      pred_x = model.dec(token_probs, as_min_encodings=True)

      loss = (init_x - pred_x).abs().mean().realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      return loss.realize()

   data = None
   s_t = time.perf_counter()
   while True:
      frames = []
      while True:
         curr_sum = sum(f.shape[0] for f in frames)
         if curr_sum >= GLOBAL_BS:
            break

         if data is None:
            datapath = next(depthpack_getter)
            print(datapath)
            if datapath is None:
               print("REACHED END OF DATA")
               return
            data = np.load(datapath)

         amnt_needed = GLOBAL_BS - curr_sum
         if data.shape[0] <= amnt_needed:
            frames.append(Tensor(data, dtype=TRAIN_DTYPE))
            data = None
         else:
            frames.append(Tensor(data[:amnt_needed], dtype=TRAIN_DTYPE))
            data = data[amnt_needed:]
      l_t = time.perf_counter()

      init_x = Tensor.cat(*frames).realize()
      assert init_x.shape[0] == GLOBAL_BS, f"{init_x.shape[0]=}, expected BS={GLOBAL_BS}"
      loss = train_step(init_x)

      step_i += 1
      e_t = time.perf_counter()
      print(f"{step_i:04d}, {(e_t-s_t)*1000:.0f} ms step ({(l_t-s_t)*1000:.0f} load, {(e_t-l_t)*1000:.0f} run), loss: {loss.item():.3f}")
      s_t = e_t

if __name__ == "__main__":
   train()
