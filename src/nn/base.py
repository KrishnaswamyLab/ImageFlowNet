import os
import torch


class BaseNetwork(torch.nn.Module):
    '''
    An base network class. For defining common utilities such as loading/saving.
    '''

    def __init__(self, **kwargs):
        super(BaseNetwork, self).__init__()
        pass

    def forward(self, *args):
        pass

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return
