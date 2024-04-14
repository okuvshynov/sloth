import os
import torch

from utils import device_map, next_id
from timers import AdditiveTimer

# let's make it forward-path only for now
class BlackboxDisk(torch.nn.Module):
    def __init__(self, module, args):
        super().__init__()
        self.module_id = next_id()
        self.compute_dtype = args.compute_dtype
        self.served_model_path = args.served_model_path
        # TODO: can we deduce this from the data itself
        self.frozen_dtype = args.frozen_dtype
        if args.init_frozen:
            torch.save(module.to('cpu').to(self.frozen_dtype), self.frozen_path())

    def frozen_path(self):
        folder = os.path.join(self.served_model_path, 'frozen')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, f'block_{self.module_id}.pt')
    
    def loaded_inner(self):
        return torch.load(self.frozen_path(), map_location='cpu')
    
    def load(self, device):
        return torch.load(self.frozen_path(), map_location=device_map(device)).to(self.compute_dtype)

    def save(self, module):
        torch.save(module.to('cpu').to(self.frozen_dtype), self.frozen_path())
    
    def forward(self, input, *args):
        device = device_map(input.device)
        with AdditiveTimer('module.load') as _:
            module = self.load(device)
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        module.eval()
        with torch.no_grad():
            return module(input, *args)