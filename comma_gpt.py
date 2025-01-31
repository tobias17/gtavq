"""
adapted from: https://github.com/commaai/commavq/blob/master/utils/gpt.py
which was adapted from https://github.com/pytorch-labs/gpt-fast
"""
from tinygrad import Tensor, nn

class CommaGptConfig:
   block_size: int = 20*129
   vocab_size: int = 1025
   n_layer: int = 24
   n_head: int = 16
   dim: int = 1024
   intermediate_size: int = 4*1024
   tokens_per_frame: int = 129

   @property
   def bos_token(self):
      return self.vocab_size - 1
   
   @property
   def head_dim(self):
      return self.dim // self.n_head

class CustomLinear:
   def __init__(self, in_dims:int, out_dims:int, bias:bool=True):
      self.weight = Tensor.zeros(in_dims, out_dims)
      self.bias   = Tensor.zeros(out_dims) if bias else None
   def __call__(self, x:Tensor) -> Tensor:
      return x.linear(self.weight, self.bias)

class Attention:
   def __init__(self, config:CommaGptConfig):
      assert config.dim % config.n_head == 0
      self.config = config
      self.c_attn = CustomLinear(config.dim, 3*config.dim, bias=True)
      self.c_proj = CustomLinear(config.dim,   config.dim, bias=True)

   def __call__(self, x:Tensor, mask:Tensor|None=None) -> Tensor:
      q,k,v = self.c_attn(x).split([self.config.dim]*3, dim=-1)
      q,k,v = [y.rearrange('B S (N D) -> B N S D', N=self.config.n_head) for y in (q,k,v)]

      y = Tensor.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=mask, is_causal=(mask is None))
      y = y.rearrange('B N S D -> B S (N D)')

      return self.c_proj(y)

class FeedForward:
   def __init__(self, config:CommaGptConfig):
      self.c_fc   = CustomLinear(config.dim, config.intermediate_size, bias=True)
      self.c_proj = CustomLinear(config.intermediate_size, config.dim, bias=True)
   
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.c_fc, Tensor.quick_gelu, self.c_proj])

class TransformerBlock:
   def __init__(self, config:CommaGptConfig):
      self.attn = Attention(config)
      self.mlp  = FeedForward(config)
      self.ln_1 = nn.LayerNorm(config.dim)
      self.ln_2 = nn.LayerNorm(config.dim)
   
   def __call__(self, x:Tensor, mask:Tensor|None=None) -> Tensor:
      h = x + self.attn(self.ln_1(x), mask)
      o = h + self.mlp(self.ln_2(h))
      return o

class Transformer:
   def __init__(self, config:CommaGptConfig):
      self.wte  = nn.Embedding(config.vocab_size, config.dim)
      self.wpe  = nn.Embedding(config.block_size, config.dim)
      self.h    = [TransformerBlock(config) for _ in range(config.n_layer)]
      self.ln_f = nn.LayerNorm(config.dim)

class GPT:
   def __init__(self, config=CommaGptConfig()):
      self.config = config
      self.transformer = Transformer(config)
      self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

   def __call__(self, idx:Tensor) -> Tensor:
      x = self.transformer.wte(idx) + self.transformer.wpe(Tensor.arange(idx.shape[1], device=idx.device))
      x = x.sequential(self.transformer.h)
      x = self.transformer.ln_f(x)
      logits = self.lm_head(x)
      return logits

   def decode_one_token(self, x:Tensor) -> Tensor:
      return self(x).argmax(axis=-1)

if __name__ == "__main__":
   from tinygrad import TinyJit, dtypes
   from tinygrad.helpers import fetch, tqdm
   from tinygrad.nn.state import load_state_dict, torch_load
   from vqvae import Decoder, transpose_and_clip
   from PIL import Image
   import numpy as np
   import os

   gpt = GPT()
   MAX_SIZE = gpt.config.block_size

   state_dict = torch_load(str(fetch("https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin?download=true", "comma_gpt.bin")))
   load_state_dict(gpt, state_dict)

   tokens = Tensor(np.load("tokens.npy").astype(np.int64)[:5])
   x_in = tokens.pad((None,(1,0)), value=MAX_SIZE).reshape(1, -1)

   @TinyJit
   def decode_step(x:Tensor) -> Tensor:
      return gpt.decode_one_token(x).realize()

   pointer = x_in.shape[1]
   x_in = x_in.pad((None,(0,MAX_SIZE-x_in.shape[1])))
   for _ in range(5):
      pad_arg = ((pointer-1,MAX_SIZE-pointer),)
      x_in = x_in * Tensor([0]).pad(pad_arg, value=1) + Tensor([gpt.config.bos_token]).pad(pad_arg)
      pointer += 1
      for _ in tqdm(range(gpt.config.tokens_per_frame - 1)):
         x_out = decode_step(x_in.realize())
         x_in[:,pointer] = x_out[:,pointer]
         pointer += 1
   print(x_in.shape)

   tokens = x_in.reshape(-1, gpt.config.tokens_per_frame)
   tokens = tokens.shrink((None,(1,gpt.config.tokens_per_frame)))

   decoder = Decoder().load_from_pretrained()
   frames = transpose_and_clip(decoder(tokens)).numpy()

   tmp_root = "/tmp/frames"
   os.makedirs(tmp_root, exist_ok=True)
   for i in tqdm(range(10)):
      Image.fromarray(frames[i]).save(f"{tmp_root}/frame{i}.png")
