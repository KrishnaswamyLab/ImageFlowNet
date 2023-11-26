import os
import torch


class BaseNetwork(torch.nn.Module):
    '''
    An base network class. For defining common utilities such as loading/saving.
    '''

    def __init__(self, **kwargs):
        super(BaseNetwork, self).__init__()
        pass

    def forward(self, *args, **kwargs):
        pass

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return

    def init_params(self):
        '''
        Parameter initialization.
        '''
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def freeze(self):
        '''
        Freeze parameters.
        '''
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        '''
        Freeze parameters.
        '''
        for p in self.parameters():
            p.requires_grad = True
