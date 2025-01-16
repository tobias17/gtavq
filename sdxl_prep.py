from datasets import load_dataset # type: ignore
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import tqdm
from vqvae import Decoder, transpose_and_clip
from PIL import Image
from threading import Thread
import os, cv2
from pathlib import Path

START_AFTER = -1
SECTION_AMOUNT = 10

OUT_ROOT = Path("./input_sdxl")
OUT_ROOT.mkdir(exist_ok=True)

DEPTHMAP_ROOT = Path("/net/tiny/raid/datasets/depthvq/depthmaps")
assert DEPTHMAP_ROOT.exists(), f"Could not find folder {DEPTHMAP_ROOT}, make sure box is mounted"

def main():
   dataset = load_dataset("/home/tobi/datasets/commavq", trust_remote_code=True)
   decoder = Decoder().load_from_pretrained()

   img_pairs = []
   index = 0
   for step in range(1, SECTION_AMOUNT + 1):
      img_pairs.append((index, index + step))
      index += step*2 + 1
   img_indices = [i for p in img_pairs for i in p]
   idx_array = np.array(img_indices)

   @TinyJit
   def decode_step(t:Tensor) -> Tensor:
      return decoder(t).realize()

   print()
   for split_key, split in dataset.items():
      if START_AFTER >= 0 and split_key in [f"{i}" for i in range(START_AFTER)]:
         continue
      print(f"Starting split '{split_key}'")
      for filepath in tqdm(split["path"]):
         out_root = Path(f"{OUT_ROOT}/{split_key}/{os.path.basename(filepath).split('.')[0]}")
         if not out_root.exists():
            os.makedirs(out_root)
         elif len(os.listdir(out_root)) >= 20:
            continue

         depthmap_root = Path(f"{DEPTHMAP_ROOT}/{split_key}/{os.path.basename(filepath).split('.')[0]}")

         depth_files = [depthmap_root/f"{i:04d}.png" for i in img_indices]
         if not all(f.exists() for f in depth_files):
            print(f"Skipping {depthmap_root}")
            continue

         tokens = np.load(filepath).astype(np.int64)
         tokens = tokens[idx_array]
         t = Tensor(tokens).reshape(len(img_indices), -1).realize()
         frames = decode_step(t)
         imgs_np = transpose_and_clip(frames).numpy()
         for i in range(len(img_pairs)):
            f1, f2 = cv2.cvtColor(imgs_np[i*2], cv2.COLOR_BGR2RGB), cv2.cvtColor(imgs_np[i*2+1], cv2.COLOR_BGR2RGB)
            mat = np.zeros((f1.shape[0]*2,f1.shape[1],f1.shape[2]))
            mat[:f1.shape[0],:,:] = f1[:,:,:]
            mat[f1.shape[0]:,:,:] = f2[:,:,:]
            cv2.imwrite(str(out_root/f"color{i+1:02d}.png"), mat)

         for i in range(len(img_pairs)):
            f1, f2 = cv2.imread(str(depth_files[i*2])), cv2.imread(str(depth_files[i*2+1]))
            mat = np.zeros((f1.shape[0]*2,f1.shape[1],f1.shape[2]))
            mat[:f1.shape[0],:,:] = f1[:,:,:]
            mat[f1.shape[0]:,:,:] = f2[:,:,:]
            cv2.imwrite(str(out_root/f"depth{i+1:02d}.png"), mat)

if __name__ == "__main__":
   main()
