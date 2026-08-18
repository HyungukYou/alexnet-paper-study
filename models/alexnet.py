import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalAlexNet(nn.Module):
    """
    Original AlexNet (Krizhevsky et al., NIPS 2012)
    Input: (B, 3, 224, 224)
    """
    def __init__(self, num_classes=10):
        super(OriginalAlexNet, self).__init__()
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        self.features = nn.Sequential(
            # Conv1: 96 kernels, 11x11, stride 4, padding 2
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            self.lrn,
            nn.MaxPool2d(kernel_size=3, stride=2), # (B, 96, 27, 27)
            
            # Conv2: 256 kernels, 5x5, padding 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            self.lrn,
            nn.MaxPool2d(kernel_size=3, stride=2), # (B, 256, 13, 13)
            
            # Conv3: 384 kernels, 3x3, padding 1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 384 kernels, 3x3, padding 1
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 256 kernels, 3x3, padding 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (B, 256, 6, 6)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TwoStreamAlexNet(nn.Module):
    """
    AlexNet Paper Figure 3 Reproduction: Two-GPU Stream Specialization Model
    - Stream 1 (GPU 1): 48 kernels specialized in color-independent grayscale edge filters
    - Stream 2 (GPU 2): 48 kernels specialized in color blobs and chrominance filters
    """
    def __init__(self, num_classes=10):
        super(TwoStreamAlexNet, self).__init__()
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # GPU 1 Stream (48 kernels) & GPU 2 Stream (48 kernels)
        self.conv1_stream1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.conv1_stream2 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        
        self._init_stream1_grayscale()
        self._init_stream2_color()
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2_stream1 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.conv2_stream2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )

    def _init_stream1_grayscale(self):
        with torch.no_grad():
            w = torch.randn(48, 1, 11, 11) * 0.05
            self.conv1_stream1.weight.copy_(w.repeat(1, 3, 1, 1))

    def _init_stream2_color(self):
        with torch.no_grad():
            w = torch.randn(48, 3, 11, 11) * 0.08
            self.conv1_stream2.weight.copy_(w)

    def forward(self, x):
        x1 = self.pool1(self.lrn(F.relu(self.conv1_stream1(x))))
        x2 = self.pool1(self.lrn(F.relu(self.conv1_stream2(x))))
        
        x1 = self.pool2(self.lrn(F.relu(self.conv2_stream1(x1))))
        x2 = self.pool2(self.lrn(F.relu(self.conv2_stream2(x2))))
        
        x_concat = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.conv3(x_concat))
        x4 = F.relu(self.conv4(x3))
        x5 = self.pool3(F.relu(self.conv5(x4)))
        
        out = torch.flatten(x5, 1)
        return self.classifier(out)
