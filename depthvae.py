from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.helpers import Context, BEAM
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load
from dataclasses import dataclass, asdict
from vqvae import Encoder, Decoder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os, time, datetime, random, json, argparse
from typing import List, Dict

# @dataclass
# class CompressorConfig:
#    in_channels:  int = 1
#    out_channels: int = 1
#    ch_mult: tuple[int,...] = (1,1,2,2,4)
#    attn_resolutions: tuple[int] = (16,)
#    resolution: int = 256
#    num_res_blocks: int = 2
#    z_channels: int = 256
#    vocab_size: int = 1024
#    ch: int = 128
#    dropout: float = 0.0

#    @property
#    def num_resolutions(self):
#       return len(self.ch_mult)

#    @property
#    def quantized_resolution(self):
#       return self.resolution // 2**(self.num_resolutions-1)

@dataclass
class CompressorConfig:
   in_channels: int = 1
   out_channels: int = 1
   ch_mult: tuple[int,...] = (1,1,2,2,4)
   attn_resolutions: tuple[int] = (16,)
   resolution: int = 256
   num_res_blocks: int = 2
   z_channels: int = 256
   vocab_size: int = 512
   ch: int = 64
   dropout: float = 0.2

   @property
   def num_resolutions(self):
      return len(self.ch_mult)

   @property
   def quantized_resolution(self):
      return self.resolution // 2**(self.num_resolutions-1)

class VQModel:
   def __init__(self, config=CompressorConfig()):
      self.enc = Encoder(config)
      self.dec = Decoder(config)
      self.dec.quantize = self.enc.quantize

__input_dims = -1
__dataset_cache: List | None = None
def get_random_batch(batch_size:int) -> np.ndarray:
   global __dataset_cache, __input_dims
   if __dataset_cache is None:
      target_shape = None
      __dataset_cache = []
      ROOT = "/raid/datasets/depthvq/depthpacks"
      for splitdir in sorted(os.listdir(ROOT)):
         for filename in sorted(os.listdir(f"{ROOT}/{splitdir}")):
            filepath = f"{ROOT}/{splitdir}/{filename}"
            if target_shape is None:
               target_shape = np.load(filepath).shape
               __input_dims = target_shape[0]
            __dataset_cache.append(np.memmap(filepath, mode="r").reshape(*target_shape))
   assert isinstance(__dataset_cache, list)
   assert __input_dims > 0
   entries = random.sample(__dataset_cache, batch_size)
   indices = np.random.randint(0, __input_dims, size=(batch_size,))
   frames = []
   for i, entry in enumerate(entries):
      frames.append(entry[indices[i]])
   return np.stack(frames)

def underscore_number(value:int) -> str:
   text = ""
   for magnitude in [1_000_000, 1_000]:
      if value >= magnitude:
         upper, value = value // magnitude, value % magnitude
         text += f"{upper}_" if len(text) == 0 else f"{upper:03d}_"
   text += f"{value}" if len(text) == 0 else f"{value:03d}"
   return text

def seed_all(seed:int):
   Tensor.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

def train():
   parser = argparse.ArgumentParser()
   parser.add_argument('-r', '--restore', type=str)
   args = parser.parse_args()

   Tensor.training = True
   seed_all(42)

   TRAIN_DTYPE = dtypes.float32
   BEAM_VALUE  = BEAM.value
   BEAM.value  = 0

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
   DEVICE_BS = 32
   GLOBAL_BS = DEVICE_BS * len(GPUS)

   AVG_EVERY  = 100
   PLOT_EVERY = 500
   EVAL_EVERY = 5000
   SAVE_EVERY = 5000

   model = VQModel()
   params = set(get_parameters(model))

   @dataclass
   class TrainInfo:
      step_i = 0
      losses = []
      prev_weights = None
      def to_json(self): return asdict(self)
      @staticmethod
      def from_json(data:Dict) -> 'TrainInfo': return TrainInfo(**data)
   
   if args.restore:
      assert os.path.exists(args.restore), f"failed to find restore root, searched for {args.restore}"
      data_filepath = os.path.join(args.restore, "data.json")
      assert os.path.exists(data_filepath), f"failed to find data.json file in restore root, searched in {args.restore}"
      with open(data_filepath, "r") as f:
         info = TrainInfo.from_json(json.load(f))
      load_state_dict(model, safe_load(info.prev_weights))
   else:
      info = TrainInfo()

   for w in params:
      w.replace(w.shard(GPUS).cast(TRAIN_DTYPE)).realize()

   LEARNING_RATE = 2**-22
   optim = AdamW(params, lr=LEARNING_RATE)

   __weights_folder = f"weights/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   def save_path(*paths:str) -> str:
      assert len(paths) > 0
      final_folder = os.path.join(__weights_folder, *paths[:-1])
      if not os.path.exists(final_folder):
         os.makedirs(final_folder)
      return os.path.join(final_folder, paths[-1])

   @TinyJit
   def train_step(init_x:Tensor) -> Tensor:
      token_probs = model.enc(init_x)
      pred_x = model.dec(token_probs, as_min_encodings=True)

      loss = (init_x - pred_x).abs().mean().realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      return loss.realize()

   eval_inputs = get_random_batch(len(GPUS))
   eval_input = Tensor(eval_inputs).shard(GPUS)
   curr_losses = []

   s_t = time.perf_counter()
   while True:
      seed_all(info.step_i)

      init_x = Tensor.cat(get_random_batch()).shard(GPUS, axis=0).realize()
      assert init_x.shape[0] == GLOBAL_BS, f"{init_x.shape[0]=}, expected BS={GLOBAL_BS}"
      l_t = time.perf_counter()

      with Context(BEAM=BEAM_VALUE):
         loss = train_step(init_x)

      info.step_i += 1
      curr_losses.append(loss.item())

      if info.step_i % AVG_EVERY == 0:
         info.losses += (sum(curr_losses) / len(curr_losses))
         curr_losses = []

      if info.step_i % PLOT_EVERY == 0:
         plt.clf()
         plt.plot(np.arange(1, len(info.losses)+1)*GLOBAL_BS, info.losses)
         plt.ylim((0,None))
         plt.title("Loss")
         fig = plt.gcf()
         fig.set_size_inches(18, 10)
         plt.savefig(save_path("graph_loss.png"))

      if info.step_i % SAVE_EVERY == 0:
         curr_weights = save_path(f"weights_{underscore_number(info.step_i)}.st")
         safe_save(get_state_dict(model), curr_weights)
         if info.prev_weights is not None:
            os.remove(info.prev_weights)
         info.prev_weights = curr_weights
         with open(save_path("data.json"), "w") as f:
            json.dump(info.to_json(), f)

      if info.step_i % EVAL_EVERY == 0:
         inputs_dirpath = save_path("evals", "input_0.png")
         if not os.path.exists(inputs_dirpath):
            for i in range(len(GPUS)):
               Image.fromarray(eval_inputs[i].reshape(*eval_inputs[i].shape[-2:])).save(save_path("evals", f"input_{i}.png"))
         token_probs = model.enc(eval_input)
         pred_x = model.dec(token_probs, as_min_encodings=True).clip(0,255).cast(dtypes.uint8).numpy()
         print(pred_x.shape)
         for i in range(len(GPUS)):
            img = pred_x[i].reshape(*pred_x[i].shape[-2:])
            Image.fromarray(img).save(save_path("evals", underscore_number(info.step_i), f"output_{i}.png"))

      e_t = time.perf_counter()
      print(f"{info.step_i:04d}, {(e_t-s_t)*1000:.0f} ms step ({(l_t-s_t)*1000:.0f} load, {(e_t-l_t)*1000:.0f} run), loss: {loss.item():.3f}")
      s_t = e_t

if __name__ == "__main__":
   train()
