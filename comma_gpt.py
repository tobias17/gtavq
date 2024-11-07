"""
adapted from: https://github.com/commaai/commavq/blob/master/utils/gpt.py
which was adapted from https://github.com/pytorch-labs/gpt-fast
"""
from tinygrad import Tensor, nn
from typing import List

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

class Attention:
   def __init__(self, config:CommaGptConfig):
      assert config.dim % config.n_head == 0
      self.config = config
      self.c_attn = nn.Linear(config.dim, 3*config.dim, bias=True)
      self.c_proj = nn.Linear(config.dim,   config.dim, bias=True)

   def __call__(self, x:Tensor, mask:Tensor|None=None) -> Tensor:
      q,k,v = self.c_attn(x).chunk(3)
      q,k,v = [y.rearrange('B S (N D) -> B N S D', N=self.config.n_head) for y in (q,k,v)]

      y = Tensor.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=mask, is_causal=(mask is None))
      y = y.rearrange('B N S D -> B S (N D)')

      return self.c_proj(y)

class FeedForward:
   def __init__(self, config:CommaGptConfig):
      self.c_fc   = nn.Linear(config.dim, config.intermediate_size, bias=True)
      self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)
   
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

   def decode_one_token(self, x:Tensor):
      return self(x).argmax(axis=-1)
