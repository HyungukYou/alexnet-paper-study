"""
===================================================================
🧪 ResNet 구현 검증 및 단위 테스트 스크립트 (test_resnet.py)
===================================================================
테스트 항목:
1. ResNet-18 / ResNet-34 / ResNet-50 텐서 순전파(Forward Pass) 및 Output Shape 검증
2. ImageNet 224x224 및 CIFAR 32x32 입력 해상도 호환성 검증
3. Backpropagation (기울기 역전파) 및 가중치 업데이트 확인
"""

import unittest
import torch
from study_resnet import ResNet18, ResNet34, ResNet50, BasicBlock, Bottleneck


class TestResNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"\n[Test Environment] Device: {self.device}")

    def test_basic_block(self):
        """BasicBlock 텐서 연산 및 Projection Shortcut 테스트"""
        x = torch.randn(2, 64, 56, 56).to(self.device)
        block_identity = BasicBlock(64, 64, stride=1).to(self.device)
        out_identity = block_identity(x)
        self.assertEqual(out_identity.shape, torch.Size([2, 64, 56, 56]))

        # Projection shortcut (stride=2, in_channels != out_channels)
        block_proj = BasicBlock(64, 128, stride=2).to(self.device)
        out_proj = block_proj(x)
        self.assertEqual(out_proj.shape, torch.Size([2, 128, 28, 28]))
        print("✅ BasicBlock (Identity & Projection Shortcut) Pass!")

    def test_bottleneck_block(self):
        """Bottleneck Block 텐서 연산 및 4배 채널 확장 테스트"""
        x = torch.randn(2, 64, 56, 56).to(self.device)
        block = Bottleneck(64, 64, stride=1).to(self.device)
        out = block(x)
        # Bottleneck expansion rate = 4 -> out channels = 256
        self.assertEqual(out.shape, torch.Size([2, 256, 56, 56]))

        # Stride 2 downsampling bottleneck
        block_down = Bottleneck(256, 128, stride=2).to(self.device)
        x_down = torch.randn(2, 256, 56, 56).to(self.device)
        out_down = block_down(x_down)
        self.assertEqual(out_down.shape, torch.Size([2, 512, 28, 28]))
        print("✅ Bottleneck Block (1x1 -> 3x3 -> 1x1 Expansion) Pass!")

    def test_resnet18_forward_and_backward(self):
        """ResNet-18 순전파 / 역전파 테스트 (224x224 ImageNet)"""
        x = torch.randn(4, 3, 224, 224).to(self.device)
        labels = torch.tensor([0, 1, 2, 3]).to(self.device)
        model = ResNet18(num_classes=10, cifar_stem=False).to(self.device)

        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 10]))

        # Backward Pass Test
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None!")
        print("✅ ResNet-18 Forward & Backward Pass Pass!")

    def test_resnet34_forward(self):
        """ResNet-34 순전파 테스트"""
        x = torch.randn(2, 3, 224, 224).to(self.device)
        model = ResNet34(num_classes=10).to(self.device)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 10]))
        print("✅ ResNet-34 Pass!")

    def test_resnet50_forward(self):
        """ResNet-50 Bottleneck 기반 순전파 테스트"""
        x = torch.randn(2, 3, 224, 224).to(self.device)
        model = ResNet50(num_classes=10).to(self.device)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 10]))
        print("✅ ResNet-50 Pass!")

    def test_cifar_stem(self):
        """CIFAR 해상도 (32x32) stem 호환성 테스트"""
        x = torch.randn(4, 3, 32, 32).to(self.device)
        model = ResNet18(num_classes=10, cifar_stem=True).to(self.device)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 10]))
        print("✅ CIFAR 32x32 Stem Pass!")


if __name__ == '__main__':
    unittest.main()
