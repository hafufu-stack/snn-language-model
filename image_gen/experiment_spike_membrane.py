"""
スパイク + 膜電位 を両方使うSNN VAE
- 通常SNNはスパイク（0/1）だけ使う
- この改良版は膜電位（連続値）も活用
- より豊かな情報伝達が可能

MNIST 28x28用、CPU実行
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from pathlib import Path
import time

spike_grad = surrogate.fast_sigmoid(slope=25)


class SpikeMembranePotentialVAE(nn.Module):
    """
    スパイク + 膜電位 を両方使うSNN VAE
    
    通常のSNN: スパイク（0 or 1）のみ使用
    この改良版: スパイク + 膜電位（連続値）を組み合わせ
    
    利点:
    - 情報量が増える（離散+連続）
    - 勾配が流れやすい（膜電位は連続なので）
    - VAEのKL Lossが学習しやすくなる可能性
    """
    def __init__(self, latent_dim=20, beta=0.9, num_steps=8, membrane_weight=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.membrane_weight = membrane_weight  # 膜電位の重み
        
        # エンコーダ
        self.enc_conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.enc_lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.enc_conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.enc_lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.enc_conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.enc_lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.pool = nn.MaxPool2d(2)
        
        # 28->14->7->3
        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)
        
        # デコーダ
        self.fc_dec = nn.Linear(latent_dim, 128 * 3 * 3)
        self.dec_lif0 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 3->6
        self.dec_lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 6->12
        self.dec_lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)   # 12->24
        self.dec_lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        self.dec_conv4 = nn.Conv2d(16, 1, 3, 1, 1)
        self.upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)
    
    def encode(self, x):
        """スパイク + 膜電位 を両方使うエンコード"""
        batch_size = x.shape[0]
        device = x.device
        
        mem1 = self.enc_lif1.init_leaky()
        mem2 = self.enc_lif2.init_leaky()
        mem3 = self.enc_lif3.init_leaky()
        
        # スパイクと膜電位の両方を蓄積
        spk_sum = torch.zeros(batch_size, 128, 3, 3, device=device)
        mem_sum = torch.zeros(batch_size, 128, 3, 3, device=device)
        
        for step in range(self.num_steps):
            inp = x * ((step + 1) / self.num_steps)
            
            h = self.enc_conv1(inp)
            spk1, mem1 = self.enc_lif1(h, mem1)
            h = self.pool(spk1)
            
            h = self.enc_conv2(h)
            spk2, mem2 = self.enc_lif2(h, mem2)
            h = self.pool(spk2)
            
            h = self.enc_conv3(h)
            spk3, mem3 = self.enc_lif3(h, mem3)
            h = self.pool(spk3)
            
            spk_sum += h
            # 膜電位も蓄積（poolingに合わせてダウンサンプル）
            mem_pooled = nn.functional.avg_pool2d(mem3, 2)
            # サイズ調整
            if mem_pooled.shape[-1] != 3:
                mem_pooled = nn.functional.adaptive_avg_pool2d(mem3, (3, 3))
            mem_sum += mem_pooled
        
        # スパイクと膜電位を組み合わせ
        # combined = (1 - w) * spike + w * membrane
        h = (1 - self.membrane_weight) * (spk_sum / self.num_steps) + \
            self.membrane_weight * torch.sigmoid(mem_sum / self.num_steps)
        
        h = h.view(batch_size, -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def decode(self, z):
        """スパイク + 膜電位 を両方使うデコード"""
        batch_size = z.shape[0]
        device = z.device
        
        mem0 = self.dec_lif0.init_leaky()
        mem1 = self.dec_lif1.init_leaky()
        mem2 = self.dec_lif2.init_leaky()
        mem3 = self.dec_lif3.init_leaky()
        
        out_sum = torch.zeros(batch_size, 1, 28, 28, device=device)
        
        for step in range(self.num_steps):
            h = self.fc_dec(z)
            spk0, mem0 = self.dec_lif0(h, mem0)
            
            # スパイク + 膜電位の組み合わせ
            h_combined = (1 - self.membrane_weight) * spk0 + \
                         self.membrane_weight * torch.sigmoid(mem0)
            h = h_combined.view(batch_size, 128, 3, 3)
            
            h = self.dec_conv1(h)
            spk1, mem1 = self.dec_lif1(h, mem1)
            h = (1 - self.membrane_weight) * spk1 + self.membrane_weight * torch.sigmoid(mem1)
            
            h = self.dec_conv2(h)
            spk2, mem2 = self.dec_lif2(h, mem2)
            h = (1 - self.membrane_weight) * spk2 + self.membrane_weight * torch.sigmoid(mem2)
            
            h = self.dec_conv3(h)
            spk3, mem3 = self.dec_lif3(h, mem3)
            h = (1 - self.membrane_weight) * spk3 + self.membrane_weight * torch.sigmoid(mem3)
            
            h = self.dec_conv4(h)
            h = self.upsample(h)
            out_sum += h
        
        return torch.sigmoid(out_sum / self.num_steps)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decode(z)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_spikes(self, x):
        """スパース性を計算（スパイクのみカウント）"""
        mem1 = self.enc_lif1.init_leaky()
        mem2 = self.enc_lif2.init_leaky()
        mem3 = self.enc_lif3.init_leaky()
        
        total_spikes = 0
        total_neurons = 0
        
        with torch.no_grad():
            for step in range(self.num_steps):
                inp = x * ((step + 1) / self.num_steps)
                h = self.enc_conv1(inp)
                spk1, mem1 = self.enc_lif1(h, mem1)
                total_spikes += spk1.sum().item()
                total_neurons += spk1.numel()
                
                h = self.pool(spk1)
                h = self.enc_conv2(h)
                spk2, mem2 = self.enc_lif2(h, mem2)
                total_spikes += spk2.sum().item()
                total_neurons += spk2.numel()
                
                h = self.pool(spk2)
                h = self.enc_conv3(h)
                spk3, mem3 = self.enc_lif3(h, mem3)
                total_spikes += spk3.sum().item()
                total_neurons += spk3.numel()
        
        return 1 - (total_spikes / total_neurons) if total_neurons > 0 else 0


def vae_loss(recon, x, mu, logvar, beta=1.0):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl, bce, kl


def train_epoch(model, loader, optimizer, device, beta_kl=1.0):
    model.train()
    total_loss = total_bce = total_kl = 0
    
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, bce, kl = vae_loss(recon, data, mu, logvar, beta_kl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_bce += bce.item()
        total_kl += kl.item()
    
    n = len(loader)
    return total_loss/n, total_bce/n, total_kl/n


def save_images(images, path, nrow=8):
    n = min(images.size(0), nrow * nrow)
    images = images[:n].cpu()
    rows = []
    for i in range(0, n, nrow):
        row = torch.cat([img.squeeze() for img in images[i:i+nrow]], dim=1)
        rows.append(row)
    grid = torch.cat(rows[:8], dim=0)
    arr = (grid.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    print("=" * 60)
    print("スパイク + 膜電位 SNN VAE 実験")
    print("=" * 60)
    
    device = 'cpu'
    print(f"Device: {device}")
    
    # データ
    transform = transforms.ToTensor()
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    print(f"訓練データ: {len(train_set)} 枚")
    
    # モデル（membrane_weight=0.3で膜電位を30%使用）
    model = SpikeMembranePotentialVAE(
        latent_dim=20, 
        beta=0.9, 
        num_steps=4,  # CPU軽量版
        membrane_weight=0.3
    ).to(device)
    
    print(f"パラメータ数: {model.count_parameters():,}")
    print(f"膜電位の重み: {model.membrane_weight}")
    
    sample, _ = next(iter(train_loader))
    sample = sample.to(device)
    sparsity = model.count_spikes(sample[:8])
    print(f"初期スパース性: {sparsity:.2%}")
    
    # 訓練
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output_dir = Path(__file__).parent / "output_spike_membrane"
    output_dir.mkdir(exist_ok=True)
    
    epochs = 10
    print(f"\n訓練開始 ({epochs} epochs)...")
    
    start = time.time()
    for epoch in range(epochs):
        beta_kl = min(1.0, 0.1 + epoch * 0.1)
        loss, bce, kl = train_epoch(model, train_loader, optimizer, device, beta_kl)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.2f} (BCE: {bce:.2f}, KL: {kl:.4f})")
    
    train_time = time.time() - start
    print(f"訓練時間: {train_time:.1f}秒")
    
    # 最終スパース性
    final_sparsity = model.count_spikes(sample[:8])
    print(f"最終スパース性: {final_sparsity:.2%}")
    
    # 画像生成
    print("\n画像生成...")
    generated = model.generate(64, device)
    save_images(generated, output_dir / "spike_membrane_generated.png")
    
    # 再構成
    model.eval()
    test_data, _ = next(iter(test_loader))
    test_data = test_data[:8].to(device)
    with torch.no_grad():
        recon, _, _ = model(test_data)
    
    top = torch.cat([test_data[i].squeeze() for i in range(8)], dim=1)
    bottom = torch.cat([recon[i].squeeze() for i in range(8)], dim=1)
    grid = torch.cat([top, bottom], dim=0).cpu()
    arr = (grid.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(output_dir / "spike_membrane_reconstruction.png")
    
    print(f"\n保存先: {output_dir}")
    print("\n【結果サマリー】")
    print(f"膜電位の重み: {model.membrane_weight}")
    print(f"スパース性: {final_sparsity:.2%}")
    print(f"訓練時間: {train_time:.1f}秒")
    print("完了！")
