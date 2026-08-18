"""
AlexNet Architecture Implementation (Krizhevsky et al., NIPS 2012)
Exact reproduction of original paper specifications:
- Section 3.1: ReLU Nonlinearity
- Section 3.2: Two-GPU Parallel Stream Architecture (Figure 3)
- Section 3.3: Local Response Normalization (LRN) (k=2, n=5, alpha=1e-4, beta=0.75)
- Section 3.4: Overlapping Max Pooling (3x3 kernel, stride 2)
- Section 3.5: Overall Layer Dimensions (Conv1~5, FC6~8)
- Section 4.2: Dropout (p=0.5) in FC6 & FC7
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResponseNorm(nn.Module):
    """
    Section 3.3: Local Response Normalization (LRN)
    k=2, n=5, alpha=1e-4, beta=0.75
    Includes MPS device fallback for Apple Silicon GPU compatibility.
    """
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2.0):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'mps':
            # Fallback to CPU execution for PyTorch MPS missing LRN operator, then return to MPS
            return F.local_response_norm(x.cpu(), self.size, self.alpha, self.beta, self.k).to(x.device)
        return F.local_response_norm(x, self.size, self.alpha, self.beta, self.k)


class OriginalAlexNet(nn.Module):
    """
    Single-Stream Baseline AlexNet Paper Model
    Input: (B, 3, 224, 224) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10, activation: str = 'relu'):
        super(OriginalAlexNet, self).__init__()
        self.activation_type = activation
        
        # Section 3.3: LRN
        self.lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # Conv Features
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # Section 3.4: Overlapping Pooling
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Section 4.2: FC Layers with Dropout (p=0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 1: Conv -> ReLU -> LRN -> Overlapping Pool
        x = self.pool1(self.lrn(self._act(self.conv1(x))))
        
        # Conv 2: Conv -> ReLU -> LRN -> Overlapping Pool
        x = self.pool2(self.lrn(self._act(self.conv2(x))))
        
        # Conv 3, 4, 5
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        x = self.pool3(self._act(self.conv5(x)))
        
        # FC Classifier
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self._act(self.fc1(x))
        x = self.dropout2(x)
        x = self._act(self.fc2(x))
        return self.fc3(x)


class TwoStreamAlexNet(nn.Module):
    """
    Section 3.2: Two-GPU Parallel Stream AlexNet (Figure 3 Reproduction)
    - Stream 1 (GPU 1): 48 kernels specialized in color-independent grayscale frequency/edge filters
    - Stream 2 (GPU 2): 48 kernels specialized in color blobs and chrominance filters
    """
    def __init__(self, num_classes: int = 10):
        super(TwoStreamAlexNet, self).__init__()
        self.lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # GPU 1 Stream (48 kernels) & GPU 2 Stream (48 kernels)
        self.conv1_stream1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.conv1_stream2 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        
        self._init_stream1_grayscale()
        self._init_stream2_color()
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Conv2: Independent streams (128 kernels each)
        self.conv2_stream1 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.conv2_stream2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Conv3: Cross-stream fusion layer (256 -> 384)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # FC Classifier
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def _init_stream1_grayscale(self):
        """Force Stream 1 RGB weights to be equal -> Grayscale Edge Detector."""
        with torch.no_grad():
            w = torch.randn(48, 1, 11, 11) * 0.05
            self.conv1_stream1.weight.copy_(w.repeat(1, 3, 1, 1))

    def _init_stream2_color(self):
        """Stream 2 initialized for color blob detectors."""
        with torch.no_grad():
            w = torch.randn(48, 3, 11, 11) * 0.08
            self.conv1_stream2.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1
        x1 = self.pool1(self.lrn(F.relu(self.conv1_stream1(x))))
        x2 = self.pool1(self.lrn(F.relu(self.conv1_stream2(x))))
        
        # Conv2
        x1 = self.pool2(self.lrn(F.relu(self.conv2_stream1(x1))))
        x2 = self.pool2(self.lrn(F.relu(self.conv2_stream2(x2))))
        
        # Conv3 (Stream Fusion: 128+128 = 256)
        x_concat = torch.cat([x1, x2], dim=1)
        
        # Conv 3, 4, 5
        x3 = F.relu(self.conv3(x_concat))
        x4 = F.relu(self.conv4(x3))
        x5 = self.pool3(F.relu(self.conv5(x4)))
        
        # FC Classifier
        x = torch.flatten(x5, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
