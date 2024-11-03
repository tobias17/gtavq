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

kill_event = Event()

def async_loader(queue:Queue, max_size:int):
   ROOT = "./depthmaps"
   for splitdir in os.listdir(ROOT):
      for scenedir in os.listdir(f"{ROOT}/{splitdir}"):
         for framename in os.listdir(f"{ROOT}/{splitdir}/{scenedir}"):
            while True:
               if kill_event.is_set():
                  return
               if queue.qsize() >= max_size:
                  time.sleep(0.05)
                  continue

               frame = np.array(Image.open(f"{ROOT}/{splitdir}/{scenedir}/{framename}"))
               queue.put(frame)
   queue.put(None)

def train():
   Tensor.training = True
   Tensor.manual_seed(42)

   model = VQModel()

   TRAIN_DTYPE = dtypes.float32
   GLOBAL_BS = 1

   LEARNING_RATE = 2**-15
   optim = AdamW(get_parameters(model), lr=LEARNING_RATE)
   step_i = 0

   queue = Queue()
   Thread(target=async_loader, args=(queue,GLOBAL_BS*4)).start()
   
   while True:
      frames = []
      while len(frames) < GLOBAL_BS:
         frame = queue.get()
         if frame is None:
            print("REACHED END OF DATA")
            return
         frames.append(Tensor(frame, dtype=TRAIN_DTYPE).unsqueeze(0))
      
      init_x = Tensor.stack(*frames)
      token_probs = model.enc(init_x)
      pred_x = model.dec(token_probs, as_min_encodings=True)

      loss = (init_x - pred_x).abs().mean()
      optim.zero_grad()
      loss.backward()
      optim.step()

      step_i += 1
      print(f"{step_i:04d}, loss: {loss.item():.3f}")

if __name__ == "__main__":
   try:
      train()
   except KeyboardInterrupt:
      kill_event.set()
