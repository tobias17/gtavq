from tinygrad import Tensor
import numpy as np
import random, os

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

def get_latest_weights_path(ext:str=".st") -> str:
   ROOT = "weights"
   folders = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]
   latest = max(folders, key=os.path.getmtime)
   files = [os.path.join(latest, f) for f in os.listdir(latest) if f.endswith(ext)]
   return max(files, key=os.path.getmtime)
