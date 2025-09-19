import timm, torch.nn as nn

class ViTSmallDR(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.resize = nn.AdaptiveAvgPool2d((224,224))
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
        self.fc = nn.Linear(384, num_classes)
    def forward(self, x):
        x = self.resize(x)
        x = self.backbone(x)
        return self.fc(x)
