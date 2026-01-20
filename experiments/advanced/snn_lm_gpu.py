"""
GPU-Accelerated Ultimate SNN using CuPy
========================================

Testing RTX 5080 for SNN acceleration!

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import time

# Try importing CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy loaded successfully!")
    print(f"   GPU: {cp.cuda.Device().name}")
    print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"❌ CuPy not available: {e}")
    print("   Falling back to NumPy...")
    import numpy as cp

import numpy as np


def ternarize(W):
    alpha = cp.mean(cp.abs(W))
    W_tern = cp.zeros_like(W)
    W_tern[W > alpha * 0.5] = 1
    W_tern[W < -alpha * 0.5] = -1
    return W_tern * alpha


class GPUUltimateSNN:
    """Ultimate SNN running on GPU via CuPy"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        cp.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = cp.random.randn(vocab_size, hidden_size).astype(cp.float32) * 0.4
        self.time_decay = cp.random.uniform(0.7, 0.95, hidden_size).astype(cp.float32)
        self.W_key = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * 0.08
        self.W_value = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * 0.08
        self.W_gate = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * 0.08
        self.W_res = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * 0.1
        self.mask = (cp.random.rand(hidden_size, hidden_size) < 0.08).astype(cp.float32)
        self.W_out = cp.random.randn(hidden_size * 2, vocab_size).astype(cp.float32) * 0.1
        self.lr = 0.12
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500)))
    
    def forward(self, sequence, time_steps=10):
        v = cp.zeros(self.hidden_size, dtype=cp.float32)
        spike_counts = cp.zeros(self.hidden_size, dtype=cp.float32)
        state = cp.zeros(self.hidden_size, dtype=cp.float32)
        
        W_res_tern = ternarize(self.W_res * self.mask)
        
        for idx, char_idx in enumerate(sequence):
            x = cp.zeros(self.vocab_size, dtype=cp.float32)
            x[char_idx] = 1.0
            
            I_in = x @ self.W_in
            mixed = I_in * (1 - self.time_decay) + state * self.time_decay
            
            key = mixed @ self.W_key
            value = mixed @ self.W_value
            gate = self.sigmoid(mixed @ self.W_gate)
            
            channel_out = gate * (key * value / (cp.abs(key).max() + 1e-10))
            state = mixed
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(cp.float32)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + channel_out * 0.5 + I_rec
                spike_counts += spiking
                v = cp.where(spiking > 0, cp.zeros_like(v), v)
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (cp.abs(v).max() + 1e-10)
        features = cp.concatenate([spike_norm, v_norm])
        
        output = features @ self.W_out
        output = output - cp.max(output)
        probs = cp.exp(output) / (cp.sum(cp.exp(output)) + 1e-10)
        
        return probs, features
    
    def train_step(self, sequence, target):
        probs, features = self.forward(sequence)
        target_vec = cp.zeros(self.vocab_size, dtype=cp.float32)
        target_vec[target] = 1.0
        self.W_out += self.lr * cp.outer(features, target_vec - probs)
        return float(-cp.log(probs[target] + 1e-10))


def prepare_data(text, seq_length=30):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    sequences, targets = [], []
    for i in range(0, len(text) - seq_length - 1, seq_length // 2):
        seq = [char_to_idx.get(c, 0) for c in text[i:i+seq_length]]
        tgt = char_to_idx.get(text[i+seq_length], 0)
        sequences.append(seq)
        targets.append(tgt)
    return sequences, targets, vocab_size


def get_text():
    text = """
    once upon a time there was a small village nestled in the mountains
    the villagers lived simple lives farming and trading with neighbors
    neural networks learn patterns from vast amounts of training data
    spiking networks mimic the brain using discrete pulses of activity
    """ * 30
    return text.lower()


def main():
    print("=" * 70)
    print("   GPU-ACCELERATED ULTIMATE SNN")
    print(f"   GPU Available: {GPU_AVAILABLE}")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    
    # Create model
    print("\n  Creating GPU model...")
    model = GPUUltimateSNN(vocab_size, hidden_size=400)
    
    # Training
    epochs = 5
    print(f"\n  Training for {epochs} epochs...")
    
    t0 = time.time()
    for epoch in range(epochs):
        losses = []
        for i in range(0, len(train_seq), 2):
            loss = model.train_step(train_seq[i], train_tgt[i])
            losses.append(loss)
        print(f"    Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")
    
    train_time = time.time() - t0
    
    # Testing
    print("\n  Testing...")
    t0 = time.time()
    test_losses = []
    for i in range(len(test_seq)):
        probs, _ = model.forward(test_seq[i])
        if GPU_AVAILABLE:
            probs = cp.asnumpy(probs)
        test_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    ppl = np.exp(np.mean(test_losses))
    test_time = time.time() - t0
    
    print("\n" + "=" * 70)
    print("   RESULTS")
    print("=" * 70)
    print(f"""
    GPU: {GPU_AVAILABLE}
    PPL: {ppl:.2f}
    Train time: {train_time:.1f}s
    Test time: {test_time:.1f}s
    """)
    
    if GPU_AVAILABLE:
        print("  🎉 GPU acceleration successful!")
    else:
        print("  ⚠️ Running on CPU (CuPy not working)")


if __name__ == "__main__":
    main()
