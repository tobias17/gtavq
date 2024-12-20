from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import prod, BEAM, Context, tqdm
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from latent_gpt import GPT
from util_ import seed_all, underscore_number, get_latest_weights_path
import json, os, math, random, time, datetime
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

DATASET_ROOT = "/raid/datasets/depthvq/latents"
class Dataset:
   data: List[List[str]]

   def __init__(self, frame_count:int):
      index_filepath = os.path.join(DATASET_ROOT, "index.json")
      amnt_per = math.ceil(frame_count / 20)
      # print(f"Amount per: {amnt_per}")
      if not os.path.exists(index_filepath):
         data = []
         for split_name in sorted(os.listdir(DATASET_ROOT)):
            split_path = os.path.join(DATASET_ROOT, split_name)
            if not os.path.isdir(split_path):
               continue
            for scene_name in sorted(os.listdir(split_path)):
               scene_path = os.path.join(split_path, scene_name)
               accum = []
               for file_name in sorted(os.listdir(scene_path)):
                  accum.append(os.path.join(scene_path, file_name))
                  if len(accum) == amnt_per:
                     data.append(accum)
                     accum = []
         random.seed(42)
         random.shuffle(data)
         with open(index_filepath, "w") as f:
            json.dump(data, f)

      self.pointer = 0
      self.frame_count = frame_count
      with open(index_filepath, "r") as f:
         self.data = json.load(f)

   def next(self, batches:int) -> Tuple[Tensor,Tensor]:
      frame_accum, depth_accum = [], []
      for _ in range(batches):
         filepaths = self.data[self.pointer]
         frames_l, depths_l = [], []
         for f in filepaths:
            d = np.load(f)
            frames_l.append(d["frames"])
            depths_l.append(d["depths"])

         frames_np = np.concatenate(frames_l)
         depths_np = np.concatenate(depths_l)
         assert frames_np.shape[0] >= self.frame_count
         assert frames_np.shape == depths_np.shape
         frame_accum.append(frames_np[:self.frame_count])
         depth_accum.append(depths_np[:self.frame_count])

         self.pointer += 1
         if self.pointer >= len(self.data):
            self.pointer = 0
      return Tensor(np.stack(frame_accum)), Tensor(np.stack(depth_accum))

def train(extra_args):
   Tensor.training = True
   seed_all(42)

   parser = argparse.ArgumentParser()
   parser.add_argument('--beam-only', action='store_true')
   args = parser.parse_args(extra_args)

   LEARNING_RATE = 2**-16
   TRAIN_DTYPE = dtypes.float32
   BEAM_VALUE  = BEAM.value
   BEAM.value  = 0

   GPUS = tuple([f"{Device.DEFAULT}:{i}" for i in range(6)])
   DEVICE_BS = 1
   GLOBAL_BS = DEVICE_BS * len(GPUS)

   AVG_EVERY  = 250
   PLOT_EVERY = 1000
   SAVE_EVERY = 5000

   model = GPT()
   params = get_parameters(model)
   print(f"Parmeters: {int(sum(prod(p.shape) for p in params) * 1e-6)}m")

   __weights_folder = f"weights/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   def save_path(*paths:str) -> str:
      assert len(paths) > 0
      final_folder = os.path.join(__weights_folder, *paths[:-1])
      if not os.path.exists(final_folder):
         os.makedirs(final_folder)
      return os.path.join(final_folder, paths[-1])

   @dataclass
   class TrainInfo:
      step_i = 0
      losses = [] # type: ignore
      prev_weights = None
      def to_json(self): return {
         "step_i":self.step_i,
         "losses":self.losses,
         "prev_weights":self.prev_weights,
      }
      @staticmethod
      def from_json(data) -> 'TrainInfo': return TrainInfo(**data)
   info = TrainInfo()

   for w in params:
      w.replace(w.shard(GPUS).cast(TRAIN_DTYPE)).realize()
   optim = AdamW(params, lr=LEARNING_RATE)

   @TinyJit
   def train_step(frames:Tensor, depths:Tensor) -> Tensor:
      pred = model(frames[:,:-1], depths[:,1:]).realize()

      loss = (pred - frames[:,1:]).square().mean().realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      return loss.realize()

   FRAMES_COUNT = model.config.max_context + 1
   dataset = Dataset(FRAMES_COUNT)
   curr_losses = []

   m = 1e3
   s_t = time.time()
   while True:
      frames, depths = dataset.next(GLOBAL_BS)
      with Context(BEAM=BEAM_VALUE):
         loss = train_step(frames.shard(GPUS, axis=0).realize(), depths.shard(GPUS, axis=0).realize())

      curr_losses.append(loss_item := loss.item())
      info.step_i += 1

      if args.beam_only:
         assert info.step_i < 10

      if info.step_i % AVG_EVERY == 0:
         info.losses.append(sum(curr_losses) / len(curr_losses))
         curr_losses = []

      if info.step_i % PLOT_EVERY == 0:
         for coll, name, ylim in ((info.losses,"Loss",(0,None)),):
            plt.clf()
            plt.plot(np.arange(1, len(coll)+1)*GLOBAL_BS*AVG_EVERY*FRAMES_COUNT, coll)
            plt.ylim(ylim)
            plt.title(name)
            fig = plt.gcf()
            fig.set_size_inches(18, 10)
            plt.savefig(save_path(f"graph_{name.lower()}.png"))

      if info.step_i % SAVE_EVERY == 0:
         curr_weights = save_path(f"weights_{underscore_number(info.step_i)}.st")
         safe_save(get_state_dict(model), curr_weights)
         if info.prev_weights is not None and os.path.exists(info.prev_weights):
            os.remove(info.prev_weights)
         info.prev_weights = curr_weights
         with open(save_path("data.json"), "w") as f:
            json.dump(info.to_json(), f)

      e_t = time.time()
      print(f"{info.step_i:04d}: {(e_t-s_t)*m:.1f} ms step, {loss_item:.4f} loss")
      s_t = e_t

def test(extra_args):
   from tinygrad.nn.state import load_state_dict, safe_load, torch_load
   from tinygrad.helpers import fetch
   from examples.stable_diffusion import StableDiffusion
   from PIL import Image

   Tensor.training = True
   Tensor.no_grad  = True
   seed_all(42)

   model = GPT()
   weights_path = get_latest_weights_path()
   print(f"Loading weights from: {weights_path}")
   load_state_dict(model, safe_load(weights_path))
   dataset = Dataset(model.config.max_context+1)
   
   stable_diffusion = StableDiffusion()
   del stable_diffusion.model
   del stable_diffusion.cond_stage_model
   weights_path = str(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))
   load_state_dict(stable_diffusion, torch_load(weights_path)['state_dict'], strict=False) # type: ignore
   def decode(x:Tensor) -> Tensor:
      B = x.shape[0]
      x = x.reshape(B, 16, 32, 4).rearrange('b h w c -> b c h w')
      x = stable_diffusion.first_stage_model.post_quant_conv(1/0.18215 * x)
      x = stable_diffusion.first_stage_model.decoder(x)
      x = (x + 1.0) / 2.0
      x = x.rearrange('b c h w -> b h w c').clip(0,1) * 255
      return x.cast(dtypes.uint8).realize()

   INPUT_SIZE = 4
   GEN_COUNT  = model.config.max_context - INPUT_SIZE

   for scene_i in range(10):
      out_folder = f"frames/scene_{scene_i}"
      os.makedirs(out_folder, exist_ok=True)

      frames, depths = dataset.next(1)
      in_frames  = decode(frames.squeeze(0)).numpy()

      curr_size = INPUT_SIZE
      z_in = frames[:,:INPUT_SIZE].realize()
      for _ in tqdm(range(GEN_COUNT)):
         z_gen = model(z_in, depths[:,1:curr_size+1]).realize()
         z_next = z_gen[:,-1:]
         z_in = z_in.cat(z_next, dim=1).realize()
         curr_size += 1
         assert z_in.shape[1] == curr_size

      out_frames = decode(z_in  .squeeze(0)).numpy()
      for i in range(INPUT_SIZE + GEN_COUNT):
         Image.fromarray( in_frames[i]).save(f"{out_folder}/real_{i:02d}.png")
         Image.fromarray(out_frames[i]).save(f"{out_folder}/gen_{i:02d}_{'r' if i < INPUT_SIZE else 'f'}.png")

if __name__ == "__main__":
   func_map = {
      "train": train,
      "test":  test,
   }

   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('func', type=str, choices=list(func_map.keys()))
   args, extra_args = parser.parse_known_args()
   func_map[args.func](extra_args)
