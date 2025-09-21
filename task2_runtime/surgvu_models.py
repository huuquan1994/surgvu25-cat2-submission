import torch
import torch.nn as nn
import torchvision

class SurgToolClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_v2_s'):
        super().__init__()
        super(SurgToolClassifier, self).__init__()

        try:
            model_loader = getattr(torchvision.models, model_name)
            self.model = model_loader()
        except AttributeError as e:
            print(e)
            raise e

        # replace final layer with num_classes=12
        num_classes = 12
        if "efficientnet_" in model_name:
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif "resnet" in model_name:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif "swin_" in model_name:
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, num_classes)
        elif "convnext_" in model_name:
            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    def load_from_ckpt(self, ckpt_path: str, device='cuda'):
        # load model from checkpoint path
        self.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)['state_dict'],
            strict=True)

    def forward(self, x):
        out = self.model(x)
        out = torch.sigmoid(out)

        return out
    
class OrganClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_v2_s'):
        super().__init__()
        super(OrganClassifier, self).__init__()

        try:
            model_loader = getattr(torchvision.models, model_name)
            self.model = model_loader()
        except AttributeError as e:
            print(e)
            raise e

        # replace final layer with num_classes=12
        num_classes = 8
        if "efficientnet_" in model_name:
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif "resnet" in model_name:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif "swin_" in model_name:
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, num_classes)
        elif "convnext_" in model_name:
            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    def load_from_ckpt(self, ckpt_path: str, device='cuda'):
        # load model from checkpoint path
        self.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)['state_dict'],
            strict=True)

    def forward(self, x):
        out = self.model(x)
        out = torch.softmax(out, dim=1)

        return out