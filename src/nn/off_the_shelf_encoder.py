import torch
import torchvision
import os
FILE_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])

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
        elif pretrained_model == 'retinal':
            self.backbone = torchvision.models.resnet50(weights=None)
            flair_model_weights = torch.load(FILE_DIR + '/external_src/FLAIR_retina/flair_resnet.pth', map_location=device)
            vision_model_weights = {}
            for key in flair_model_weights.keys():
                if 'vision_model' in key:
                    vision_model_weights[key.replace('vision_model.model.', '')] = flair_model_weights[key]
            self.backbone.load_state_dict(vision_model_weights, strict=False)
            self.backbone.fc = torch.nn.Identity()
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
