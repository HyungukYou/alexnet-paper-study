import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class AlexNetActivationExp(nn.Module):
    def __init__(self, activation='relu', num_classes=10):
        super(AlexNetActivationExp, self).__init__()
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
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        )

    def _act(self, x):
        if self.activation_type == 'relu':
            return F.relu(x)
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
        feats = self.classifier[:2](x)
        feats = self._act(feats)
        feats = self.classifier[2:4](feats)
        feats = self._act(feats)
        return self.classifier[4](feats)

def run_activation_experiment(trainloader, device, save_path="act_graph.png", epochs=2):
    def run_one(act_name):
        print(f"🏋️ [{act_name.upper()}] 기반 AlexNet 학습 시작...")
        model = AlexNetActivationExp(activation=act_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        
        losses = []
        start_t = time.time()
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 15 == 14:
                    losses.append(running_loss / 15)
                    running_loss = 0.0
                    
        elapsed = time.time() - start_t
        print(f"✅ [{act_name.upper()}] 완료! (소요 시간: {elapsed:.1f}초)")
        return losses, elapsed

    l_relu, t_relu = run_one('relu')
    l_tanh, t_tanh = run_one('tanh')
    l_sig, t_sig = run_one('sigmoid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(l_relu, label='ReLU (Paper Baseline: Fast Convergence)', color='crimson', linewidth=2.8)
    ax1.plot(l_tanh, label='Tanh (Saturating Nonlinearity)', color='royalblue', linewidth=2.5, linestyle='--')
    ax1.plot(l_sig, label='Sigmoid (Severe Vanishing Gradient)', color='orange', linewidth=2.5, linestyle=':')

    ax1.set_title('AlexNet Paper Section 3.1 & Figure 1: Loss Convergence', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Training Iterations (x15 batches)', fontsize=11)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    acts = ['ReLU', 'Tanh', 'Sigmoid']
    times = [t_relu, t_tanh, t_sig]
    colors = ['crimson', 'royalblue', 'orange']

    bars = ax2.bar(acts, times, color=colors, width=0.45, alpha=0.85)
    ax2.set_title('Total Training Time (2 Epochs)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Elapsed Time (Seconds)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}s", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('AlexNet Section 3.1 Experimental Comparison (ReLU vs Tanh vs Sigmoid)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"🎉 비교 그래프가 {save_path} 에 저장되었습니다!")
