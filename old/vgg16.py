from tinygrad import Tensor, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, torch_load
from functools import partial

class vgg16:
   def __init__(self):
      self.features = [
         nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         partial(Tensor.max_pool2d, kernel_size=2, stride=2, padding=0, dilation=1),
         nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         partial(Tensor.max_pool2d, kernel_size=2, stride=2, padding=0, dilation=1),
         nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         partial(Tensor.max_pool2d, kernel_size=2, stride=2, padding=0, dilation=1),
         nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         partial(Tensor.max_pool2d, kernel_size=2, stride=2, padding=0, dilation=1),
         nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         Tensor.relu,
         partial(Tensor.max_pool2d, kernel_size=2, stride=2, padding=0, dilation=1),
      ]
      self.classifier = [
         nn.Linear(in_features=25088, out_features=4096, bias=True),
         Tensor.relu,
         partial(Tensor.dropout, p=0.5),
         nn.Linear(in_features=4096, out_features=4096, bias=True),
         Tensor.relu,
         partial(Tensor.dropout, p=0.5),
         nn.Linear(in_features=4096, out_features=1000, bias=True),
      ]
   
   def load_from_pretrained(self, url:str="https://download.pytorch.org/models/vgg16-397923af.pth", name:str="vgg16.pth") -> 'vgg16':
      load_state_dict(self, torch_load(str(fetch(url, name))))
      return self

if __name__ == "__main__":
   vgg16().load_from_pretrained()
