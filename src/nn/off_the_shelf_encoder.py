import torch
import torchvision


class VisionEncoder():
    def __init__(self,
                 pretrained_model: str,
                 device: torch.device):
        if pretrained_model == 'resnet18':
            self.backbone = torchvision.models.resnet18(weights='DEFAULT')
            self.backbone.fc = torch.nn.Identity()
        elif pretrained_model == 'convnext_tiny':
            self.backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
            self.backbone.classifier[-1] = torch.nn.Identity()
        elif pretrained_model == 'mobilenetv3_small':
            self.backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            self.backbone.classifier[-1] = torch.nn.Identity()
        self.backbone.eval()
        self.backbone.to(device)
        self.device = device

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4
        assert image.shape[1] in [1, 3]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        latent_embedding = self.backbone(image.float().to(self.device))
        return latent_embedding
