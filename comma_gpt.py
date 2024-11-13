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
   from tinygrad.helpers import fetch, tqdm
   from tinygrad.nn.state import load_state_dict, torch_load
   from vqvae import Decoder, transpose_and_clip
   from PIL import Image
   import numpy as np
   import os

   gpt = GPT()

   state_dict = torch_load(str(fetch("https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin?download=true", "comma_gpt.bin")))
   # for k, w in state_dict.items():
   #    if k.endswith(".c_proj.weight") or k.endswith(".c_attn.weight") or k.endswith(".c_fc.weight"):
   #       state_dict[k] = w.to(Device.DEFAULT).T
   load_state_dict(gpt, state_dict)

   tokens  = np.load("tokens.npy").astype(np.int64)

   x_in = Tensor(tokens[:5])
   for _ in tqdm(range(5)):
      x_out = gpt.decode_one_token(x_in)
      x_in = x_in.cat(x_out[-1:])
   print(x_in.shape)

   decoder = Decoder().load_from_pretrained()
   frames = transpose_and_clip(decoder(x_in)).numpy()
   
   tmp_root = "/tmp/frames"
   os.makedirs(tmp_root, exist_ok=True)
   for i in range(frames.shape[0]):
      Image.fromarray(frames[i]).save(f"{tmp_root}/frame{i}.png")
