import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalAlexNet(nn.Module):
    """
    Original AlexNet (Krizhevsky et al., NIPS 2012)
    Input: (B, 3, 224, 224)
    Supports configurable activation functions ('relu', 'tanh', 'sigmoid') for Section 3.1 experiments.
    """
    def __init__(self, num_classes=10, activation='relu'):
        super(OriginalAlexNet, self).__init__()
        self.activation_type = activation
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def _act(self, x):
        if self.activation_type == 'relu':
            return F.relu(x, inplace=True)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.pool1(self.lrn(self._act(self.conv1(x))))
        x = self.pool2(self.lrn(self._act(self.conv2(x))))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        x = self.pool3(self._act(self.conv5(x)))
        
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self._act(self.fc1(x))
        x = self.dropout2(x)
        x = self._act(self.fc2(x))
        return self.fc3(x)


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
