from datasets import load_dataset # type: ignore
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import tqdm, fetch, Context
from tinygrad.nn.state import load_state_dict, torch_load
from examples.stable_diffusion import StableDiffusion
from vqvae import Decoder
from PIL import Image
import numpy as np
import os

START_AFTER = -1
BATCH_SIZE = 20
AMOUNT_PER = 240

DATASET_ROOT  = "/net/tiny/raid/datasets/depthvq"
DEPTHMAP_ROOT = f"{DATASET_ROOT}/depthmaps"
LATENTS_ROOT  = f"{DATASET_ROOT}/latents"

def main():
   assert os.path.exists(DEPTHMAP_ROOT)
   if not os.path.exists(LATENTS_ROOT):
      os.mkdir(LATENTS_ROOT)

   model = StableDiffusion()
   load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False) # type: ignore

   decoder = Decoder().load_from_pretrained()

   @TinyJit
   def encode_latent(im:Tensor) -> Tensor:
      x = im.div(255.0) * 2.0 - 1.0 
      z = model.first_stage_model.quant_conv(model.first_stage_model.encoder(x)).chunk(2, dim=1)[0]
      return z.mul(0.18215).realize()

   dataset = load_dataset("/home/tobi/datasets/commavq", trust_remote_code=True)
   for split_key, split in dataset.items():
      if START_AFTER >= 0 and split_key in [f"{i}" for i in range(START_AFTER)]:
         continue
      print(f"Starting split '{split_key}'")
      for filepath in tqdm(split["path"]):
         filename = os.path.basename(filepath)
         scene_folder = filename.split(".")[0]
         all_tokens = None

         start_i = 0
         while True:
            latents_path = os.path.join(LATENTS_ROOT, split_key, scene_folder, f"latent_{start_i}.npz")
            if not os.path.exists(latents_path):
               depthmaps = []
               for frame_i in range(BATCH_SIZE):
                  depthmap_path = os.path.join(DEPTHMAP_ROOT, split_key, scene_folder, f"{start_i+frame_i:04d}.png")
                  if not os.path.exists(depthmap_path):
                     break
                  try:
                     img_gs = Image.open(depthmap_path).convert('RGB')
                     img_rgb = Image.new('RGB', img_gs.size)
                     img_rgb.paste(img_gs)

                     im = Tensor(np.array(img_rgb)).cast(dtypes.float32).unsqueeze(0).rearrange('b h w c -> b c h w')
                     depthmaps.append(im.realize())
                  except Exception as ex:
                     print(f"Ran into error loading {depthmap_path}: {ex}")
                     break
               else:
                  if all_tokens is None:
                     all_tokens = np.load(filepath)
                  assert isinstance(all_tokens, np.ndarray)
                  tokens = Tensor(all_tokens[start_i:start_i+BATCH_SIZE]).rearrange('b h w -> b (h w)')

                  frames = decoder(tokens)
                  depths = Tensor.cat(*depthmaps)
                  assert depths.shape == frames.shape, f"shape mismatch, {depths.shape} != {frames.shape}"

                  with Context(BEAM=1):
                     frames_z = encode_latent(frames.realize()).cast(dtypes.float16).numpy()
                     depths_z = encode_latent(depths.realize()).cast(dtypes.float16).numpy()

                  assert depths_z.shape == frames_z.shape, f"shape mismatch, {depths_z.shape} != {frames_z.shape}"
                  os.makedirs(os.path.dirname(latents_path), exist_ok=True)
                  np.savez(latents_path, frames=frames_z, depths=depths_z)
            
            start_i += BATCH_SIZE
            if start_i >= AMOUNT_PER:
               break

if __name__ == "__main__":
   main()
