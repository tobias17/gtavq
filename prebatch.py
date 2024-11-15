from datasets import load_dataset # type: ignore
from tinygrad.helpers import tqdm
from taming_vqgan import load_taming_vqgan, to_input
from PIL import Image
import numpy as np
import os
import torch

BATCH_SIZE = 20
AMOUNT_PER = 240

DATASET_ROOT  = "/net/tiny/raid/datasets/depthvq"
DEPTHMAP_ROOT = f"{DATASET_ROOT}/depthmaps"
BATCHED_ROOT  = f"{DATASET_ROOT}/batched"

def main():
   assert os.path.exists(DEPTHMAP_ROOT)
   if not os.path.exists(BATCHED_ROOT):
      os.mkdir(BATCHED_ROOT)

   model = load_taming_vqgan().cuda().eval()

   dataset = load_dataset("/home/tobi/datasets/commavq", trust_remote_code=True)
   for split_key, split in dataset.items():
      print((banner := "\n"+"="*80+"\n\n") + f"Starting split '{split_key}'" + banner[::-1])
      for filepath in tqdm(split["path"]):
         filename = os.path.basename(filepath)
         scene_folder = filename.split(".")[0]
         all_frames = None

         start_i = 0
         while True:
            batched_path = os.path.join(BATCHED_ROOT, split_key, scene_folder, f"batch_{start_i}.npz")
            if not os.path.exists(batched_path):
               depthmaps = []
               for frame_i in range(BATCH_SIZE):
                  depthmap_path = os.path.join(DEPTHMAP_ROOT, split_key, scene_folder, f"{start_i+frame_i:04d}.png")
                  if not os.path.exists(depthmap_path):
                     break
                  depthmaps.append(to_input(Image.open(depthmap_path)).cuda())
               else:
                  x_in = torch.cat(depthmaps)
                  with torch.no_grad():
                     _, _, (_, _, min_indices) = model.encode(x_in)
                  depths = min_indices.reshape(BATCH_SIZE, 8, 16).cpu().numpy()
                  assert isinstance(depths, np.ndarray)

                  if all_frames is None:
                     all_frames = np.load(filepath)
                  assert isinstance(all_frames, np.ndarray)
                  frames = all_frames[start_i:start_i+BATCH_SIZE]

                  assert depths.shape == frames.shape, f"shape mismatch, {depths.shape} != {frames.shape}"
                  os.makedirs(os.path.dirname(batched_path), exist_ok=True)
                  np.savez(batched_path, frames=frames, depths=depths)
            
            start_i += BATCH_SIZE
            if start_i >= AMOUNT_PER:
               break

if __name__ == "__main__":
   main()
