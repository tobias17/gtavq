from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import prod, BEAM, Context
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from depth_gpt import GPT
from util import seed_all, underscore_number
import json, os, math, random, time, datetime
import numpy as np
from typing import List
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

   def next(self, batches:int):
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

def train():
   Tensor.training = True
   seed_all(42)

   LEARNING_RATE = 2**-18
   TRAIN_DTYPE = dtypes.float32
   BEAM_VALUE  = BEAM.value
   BEAM.value  = 0

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
   DEVICE_BS = 6
   GLOBAL_BS = DEVICE_BS * len(GPUS)

   AVG_EVERY  = 100
   PLOT_EVERY = 500
   SAVE_EVERY = 10000

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
      logits = model(frames[:,:-1], depths[:,1:]).realize()

      loss = logits.sparse_categorical_crossentropy(frames[:,1:]).realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      return loss.realize()

   dataset = Dataset(model.config.max_context+1)
   curr_losses = []

   m = 1e3
   s_t = time.time()
   while True:
      frames, depths = dataset.next(GLOBAL_BS)
      with Context(BEAM=BEAM_VALUE):
         loss = train_step(frames.shard(GPUS, axis=0).realize(), depths.shard(GPUS, axis=0).realize())

      curr_losses.append(loss_item := loss.item())
      info.step_i += 1

      if info.step_i % AVG_EVERY == 0:
         info.losses.append(sum(curr_losses) / len(curr_losses))
         curr_losses = []

      if info.step_i % PLOT_EVERY == 0:
         plt.clf()
         plt.plot(np.arange(1, len(info.losses)+1)*GLOBAL_BS*AVG_EVERY, info.losses)
         plt.ylim((0,None))
         plt.title("Loss")
         fig = plt.gcf()
         fig.set_size_inches(18, 10)
         plt.savefig(save_path(f"graph_loss.png"))

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


if __name__ == "__main__":
   train()
