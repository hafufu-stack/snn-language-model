"""
膜電位の重み比較実験
- 50%版と70%版を連続実行
- 0%（昨日）、30%（さっき）と比較用
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
    def __init__(self, latent_dim=20, beta=0.9, num_steps=4, membrane_weight=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.membrane_weight = membrane_weight
        
        self.enc_conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.enc_lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.enc_lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.enc_lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.pool = nn.MaxPool2d(2)
        
        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim, 128 * 3 * 3)
        self.dec_lif0 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.dec_lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.dec_conv4 = nn.Conv2d(16, 1, 3, 1, 1)
        self.upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)
    
    def encode(self, x):
        batch_size = x.shape[0]
        device = x.device
        mem1 = self.enc_lif1.init_leaky()
        mem2 = self.enc_lif2.init_leaky()
        mem3 = self.enc_lif3.init_leaky()
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
            mem_pooled = nn.functional.adaptive_avg_pool2d(mem3, (3, 3))
            mem_sum += mem_pooled
        
        w = self.membrane_weight
        h = (1 - w) * (spk_sum / self.num_steps) + w * torch.sigmoid(mem_sum / self.num_steps)
        h = h.view(batch_size, -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        batch_size = z.shape[0]
        device = z.device
        mem0 = self.dec_lif0.init_leaky()
        mem1 = self.dec_lif1.init_leaky()
        mem2 = self.dec_lif2.init_leaky()
        mem3 = self.dec_lif3.init_leaky()
        out_sum = torch.zeros(batch_size, 1, 28, 28, device=device)
        w = self.membrane_weight
        
        for step in range(self.num_steps):
            h = self.fc_dec(z)
            spk0, mem0 = self.dec_lif0(h, mem0)
            h = (1 - w) * spk0 + w * torch.sigmoid(mem0)
            h = h.view(batch_size, 128, 3, 3)
            h = self.dec_conv1(h)
            spk1, mem1 = self.dec_lif1(h, mem1)
            h = (1 - w) * spk1 + w * torch.sigmoid(mem1)
            h = self.dec_conv2(h)
            spk2, mem2 = self.dec_lif2(h, mem2)
            h = (1 - w) * spk2 + w * torch.sigmoid(mem2)
            h = self.dec_conv3(h)
            spk3, mem3 = self.dec_lif3(h, mem3)
            h = (1 - w) * spk3 + w * torch.sigmoid(mem3)
            h = self.dec_conv4(h)
            h = self.upsample(h)
            out_sum += h
        return torch.sigmoid(out_sum / self.num_steps)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate(self, n, device):
        z = torch.randn(n, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decode(z)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_spikes(self, x):
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


def train_model(model, loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss = total_kl = 0
        for data, _ in loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            beta_kl = min(1.0, 0.1 + epoch * 0.1)
            loss, bce, kl = vae_loss(recon, data, mu, logvar, beta_kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_kl += kl.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.2f}, KL: {total_kl/len(loader):.4f}")
    return total_loss / len(loader), total_kl / len(loader)


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
    print("膜電位の重み比較実験 (50% vs 70%)")
    print("=" * 60)
    
    device = 'cpu'
    epochs = 10
    
    transform = transforms.ToTensor()
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    print(f"訓練データ: {len(train_set)} 枚\n")
    
    output_dir = Path(__file__).parent / "output_membrane_comparison"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    weights_to_test = [0.5, 0.7]
    
    for weight in weights_to_test:
        print("=" * 50)
        print(f"【膜電位の重み: {int(weight*100)}%】")
        print("=" * 50)
        
        model = SpikeMembranePotentialVAE(
            latent_dim=20, num_steps=4, membrane_weight=weight
        ).to(device)
        
        sample, _ = next(iter(train_loader))
        sample = sample.to(device)
        init_sparsity = model.count_spikes(sample[:8])
        print(f"初期スパース性: {init_sparsity:.2%}")
        
        start = time.time()
        final_loss, final_kl = train_model(model, train_loader, epochs, device)
        train_time = time.time() - start
        
        final_sparsity = model.count_spikes(sample[:8])
        print(f"最終スパース性: {final_sparsity:.2%}")
        print(f"訓練時間: {train_time:.1f}秒")
        
        # 画像保存
        generated = model.generate(64, device)
        save_images(generated, output_dir / f"generated_{int(weight*100)}pct.png")
        
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
        Image.fromarray(arr).save(output_dir / f"reconstruction_{int(weight*100)}pct.png")
        
        results.append({
            'weight': weight,
            'loss': final_loss,
            'kl': final_kl,
            'sparsity': final_sparsity,
            'time': train_time
        })
        print()
    
    # 結果比較
    print("\n" + "=" * 60)
    print("【全結果比較】")
    print("=" * 60)
    print(f"{'膜電位重み':<12} {'Loss':<12} {'KL Loss':<12} {'スパース性':<12}")
    print("-" * 48)
    
    # 過去の結果も追加
    print(f"{'0% (昨日)':<12} {'206.61':<12} {'0.0000':<12} {'99.17%':<12}")
    print(f"{'30% (さっき)':<12} {'111.17':<12} {'0.6332':<12} {'97.95%':<12}")
    
    for r in results:
        print(f"{int(r['weight']*100)}%{'':<10} {r['loss']:<12.2f} {r['kl']:<12.4f} {r['sparsity']:<12.2%}")
    
    print(f"\n画像は {output_dir} に保存されました")
    print("完了！")
