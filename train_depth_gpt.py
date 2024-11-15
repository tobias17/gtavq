from tinygrad import Tensor
from tinygrad.helpers import prod
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from depth_gpt import GPT
import json, os, math, random
import numpy as np
from typing import List

DATASET_ROOT = "/raid/datasets/depthvq/batched"
class Dataset:
   data: List[List[str]]

   def __init__(self, frame_count:int):
      index_filepath = os.path.join(DATASET_ROOT, "index.json")
      amnt_per = math.ceil(frame_count / 20)
      print(f"Amount per: {amnt_per}")
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
         frame_accum.append(frames_np)
         depth_accum.append(depths_np)

         self.pointer += 1
         if self.pointer >= len(self.data):
            self.pointer = 0
      return Tensor(np.stack(frame_accum)), Tensor(np.stack(depth_accum))

def seed_all(seed:int):
   Tensor.manual_seed(seed)
   np.random.seed(seed)

def train():
   seed_all(42)

   GLOBAL_BS = 4
   LR = 2**-16

   model = GPT()
   params = get_parameters(model)
   optim = AdamW(params)

   dataset = Dataset(model.config.max_context+1)

   print(f"Parmeters: {int(sum(prod(p.shape) for p in params) * 1e-6)}m")

   for _ in range(1):
      frames, depths = dataset.next(GLOBAL_BS)
      print(frames.shape)
      print(depths.shape)


   # logits = model(Tensor.randint(2,10,128,high=128), Tensor.randint(2,10,128,high=128)).realize()


if __name__ == "__main__":
   train()
