import timm, torch.nn as nn, torch.nn.functional as F

class RSGRes34DR(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.feat = timm.create_model("resnet34", pretrained=True, features_only=True, out_indices=(4,))
        self.attn = nn.Sequential(nn.Conv2d(512,128,1), nn.ReLU(inplace=True),
                                  nn.Conv2d(128,1,1), nn.Sigmoid())
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        f = self.feat(x)[0]
        f = f * self.attn(f)
        f = F.adaptive_avg_pool2d(f,1).flatten(1)
        return self.fc(f)
