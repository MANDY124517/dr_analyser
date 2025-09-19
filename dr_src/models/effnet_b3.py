import timm, torch.nn as nn

class EffNetB3DR(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
        self.fc = nn.Sequential(nn.Dropout(0.4),
                                nn.Linear(self.backbone.num_features, num_classes))
    def forward(self, x):
        return self.fc(self.backbone(x))
