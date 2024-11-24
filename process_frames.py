from tinygrad.helpers import tqdm
from PIL import Image
import numpy as np
import os, sys

GLOBAL_ROOT = "./frames/game_capture"

def crop_images():
   INPUT_ROOT = f"{GLOBAL_ROOT}/raw"
   OUTPUT_ROOT = f"{GLOBAL_ROOT}/cropped"
   if not os.path.exists(OUTPUT_ROOT):
      os.mkdir(OUTPUT_ROOT)
   
   print("Cropping Images")
   for filename in tqdm(os.listdir(INPUT_ROOT)):
      if not os.path.exists(out_path := f"{OUTPUT_ROOT}/{filename}"):
         img = np.array(Image.open(f"{INPUT_ROOT}/{filename}"))
         img = img[40:-5, 5:-5]

         h, w, _ = img.shape
         target_height = w // 2
         assert target_height <= h, f"{target_height} > {h}"
         y_start = (h - target_height) // 2
         img = img[y_start:y_start+target_height]
         img = Image.fromarray(img).resize((256,128))
         img.save(out_path)

def make_depthmaps():
   INPUT_ROOT = f"{GLOBAL_ROOT}/cropped"
   OUTPUT_ROOT = f"{GLOBAL_ROOT}/depth"
   if not os.path.exists(OUTPUT_ROOT):
      os.mkdir(OUTPUT_ROOT)

   files_to_run = []
   all_filenames = os.listdir(INPUT_ROOT)
   for filename in all_filenames:
      if not os.path.exists(f"{OUTPUT_ROOT}/{filename}"):
         files_to_run.append(filename)
   
   print(f"Found {len(files_to_run)}/{len(all_filenames)} depthmaps needing to be run")
   if len(files_to_run) == 0:
      return

   sys.path.append("../depth-fm")
   from depthfm import DepthFM # type: ignore
   import torch
   model = DepthFM("../depth-fm/checkpoints/depthfm-v1.ckpt")
   model.cuda().eval()
   model.model.dtype = torch.float16
   for filename in tqdm(files_to_run):
      frame = np.array(Image.open(f"{INPUT_ROOT}/{filename}")) / 127.5 - 1
      im = torch.from_numpy(frame.astype(np.float16)).cuda().permute(2, 0, 1)
      im = im.reshape(1, *im.shape)
      with torch.autocast(device_type="cuda", dtype=torch.half):
         depth = model.predict_depth(im, num_steps=1, ensemble_size=0)
      depth = depth.squeeze(1).cpu().numpy()
      depth = (depth * 255).astype(np.uint8)
      Image.fromarray(depth[0]).save(f"{OUTPUT_ROOT}/{filename}")

def tokenize_depthmaps():
   INPUT_ROOT = f"{GLOBAL_ROOT}/depth"
   OUTPUT_FILE = f"{GLOBAL_ROOT}/depth_tokens.npy"
   if os.path.exists(OUTPUT_FILE):
      print("Depth tokens already exist, skipping")
      return
   
   filenames = sorted(os.listdir(INPUT_ROOT))
   
   BATCH_SIZE = 20
   import torch
   from taming_vqgan import load_taming_vqgan, to_input
   model = load_taming_vqgan().cuda().eval()

   def run_batch(images):
      x_in = torch.cat(images)
      with torch.no_grad():
         _, _, (_, _, min_indices) = model.encode(x_in)
      return min_indices.reshape(BATCH_SIZE, 8, 16).cpu().numpy()

   x_out = []
   batch = []
   for filename in tqdm(filenames):
      depthmap_path = f"{INPUT_ROOT}/{filename}"
      batch.append(to_input(Image.open(depthmap_path)).cuda())
      if len(batch) >= BATCH_SIZE:
         x_out.append(run_batch(batch))
         batch = []
   if len(batch) > 0:
      x_out.append(run_batch(batch))
   
   output = np.concatenate(x_out, axis=0)
   output = output.reshape(-1, 128)
   print(f"Created depth tokens with shape: {output.shape}")
   assert output.shape[0] == len(filenames), f"{output.shape[0]} != {len(filenames)}"
   np.save(OUTPUT_FILE, output)

def tokenize_frames():
   INPUT_ROOT = f"{GLOBAL_ROOT}/cropped"
   OUTPUT_FILE = f"{GLOBAL_ROOT}/frame_tokens.npy"
   if os.path.exists(OUTPUT_FILE):
      print("Frame tokens already exist, skipping")
      return
   
   filenames = sorted(os.listdir(INPUT_ROOT))
   
   BATCH_SIZE = 20
   from tinygrad import Tensor
   from vqvae import Encoder
   encoder = Encoder().load_from_pretrained()

   def run_batch(images):
      x_in = Tensor.stack(*images)
      return encoder(x_in).argmax(axis=-1).numpy()

   x_out = []
   batch = []
   for filename in tqdm(filenames):
      frame_path = f"{INPUT_ROOT}/{filename}"
      frame = np.array(Image.open(frame_path)).astype(np.float16)
      batch.append(Tensor(frame).permute(2,0,1).realize())
      if len(batch) >= BATCH_SIZE:
         x_out.append(run_batch(batch))
         batch = []
   if len(batch) > 0:
      x_out.append(run_batch(batch))
   
   output = np.concatenate(x_out, axis=0)
   print(f"Created frame tokens with shape: {output.shape}")
   assert output.shape[0] == len(filenames), f"{output.shape[0]} != {len(filenames)}"
   np.save(OUTPUT_FILE, output)

def gen_images():
   depths = np.load(f"{GLOBAL_ROOT}/depth_tokens.npy")
   frames = np.load(f"{GLOBAL_ROOT}/frame_tokens.npy")

if __name__ == "__main__":
   crop_images()
   make_depthmaps()
   tokenize_depthmaps()
   tokenize_frames()
   gen_images()
