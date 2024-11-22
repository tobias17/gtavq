from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import prod, BEAM, Context, tqdm
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from depth_gpt import GPT
from util import seed_all, underscore_number, get_latest_weights_path
import json, os, math, random, time, datetime
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

DATASET_ROOT = "/raid/datasets/depthvq/batched"
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
            frames_l.append(d["frames"].reshape(20, 128))
            depths_l.append(d["depths"].reshape(20, 128))

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

   LEARNING_RATE = 2**-14
   TRAIN_DTYPE = dtypes.float32
   BEAM_VALUE  = BEAM.value
   BEAM.value  = 0

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
   DEVICE_BS = 4
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
      losses = []
      acc = []
      prev_weights = None
      def to_json(self): return {
         "step_i":self.step_i,
         "losses":self.losses,
         "acc": self.acc,
         "prev_weights":self.prev_weights,
      }
      @staticmethod
      def from_json(data) -> 'TrainInfo': return TrainInfo(**data)
   info = TrainInfo()

   for w in params:
      w.replace(w.shard(GPUS).cast(TRAIN_DTYPE)).realize()
   optim = AdamW(params, lr=LEARNING_RATE)

   @TinyJit
   def train_step(frames:Tensor, depths:Tensor) -> Tuple[Tensor,Tensor]:
      logits = model(frames[:,:-1], depths[:,1:]).realize()

      loss = logits.sparse_categorical_crossentropy(frames[:,1:]).realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      acc = (logits.argmax(axis=-1) == frames[:,1:]).cast(dtypes.float32).mul(100.0).mean()
      return loss.realize(), acc.realize()

   FRAMES_COUNT = model.config.max_context + 1
   dataset = Dataset(FRAMES_COUNT)
   curr_losses, curr_acc = [], []

   m = 1e3
   s_t = time.time()
   while True:
      frames, depths = dataset.next(GLOBAL_BS)
      with Context(BEAM=BEAM_VALUE):
         loss, acc = train_step(frames.shard(GPUS, axis=0).realize(), depths.shard(GPUS, axis=0).realize())

      curr_losses.append(loss_item := loss.item())
      curr_acc.append(acc_item := acc.item())
      info.step_i += 1

      if args.beam_only:
         assert info.step_i < 10

      if info.step_i % AVG_EVERY == 0:
         info.losses.append(sum(curr_losses) / len(curr_losses))
         curr_losses = []
         info.acc.append(sum(curr_acc) / len(curr_acc))
         curr_acc = []

      if info.step_i % PLOT_EVERY == 0:
         for coll, name, ylim in ((info.losses,"Loss",(0,None)), (info.acc,"Acc",(0,100))):
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
      print(f"{info.step_i:04d}: {(e_t-s_t)*m:.1f} ms step, {loss_item:.4f} loss, {acc_item:.1f}% acc")
      s_t = e_t

def test(extra_args):
   from tinygrad.nn.state import safe_load, load_state_dict
   from vqvae import Decoder, transpose_and_clip
   from PIL import Image

   Tensor.training = True
   Tensor.no_grad  = True
   seed_all(42)

   model = GPT()
   weights_path = get_latest_weights_path()
   print(f"Loading weights from: {weights_path}")
   load_state_dict(model, safe_load(weights_path))
   dataset = Dataset(model.config.max_context+1)
   decoder = Decoder().load_from_pretrained()

   INPUT_SIZE = 4
   GEN_COUNT  = model.config.max_context - INPUT_SIZE

   for scene_i in range(10):
      out_folder = f"frames/scene_{scene_i}"
      os.makedirs(out_folder, exist_ok=True)

      frames, depths = dataset.next(1)

      curr_size = INPUT_SIZE
      x_in = frames[:,:INPUT_SIZE]
      for _ in tqdm(range(GEN_COUNT)):
         logits = model(x_in, depths[:,1:curr_size+1]).realize()
         next_frame = logits[:,-1:].argmax(axis=-1)
         x_in = x_in.cat(next_frame, dim=1).realize()
         curr_size += 1
         assert x_in.shape[1] == curr_size

      in_frames  = transpose_and_clip(decoder(frames.squeeze(0))).numpy()
      out_frames = transpose_and_clip(decoder(x_in.squeeze(0))).numpy()
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
