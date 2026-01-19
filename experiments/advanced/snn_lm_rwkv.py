"""
RWKV-Inspired SNN: Gated Recurrent Spiking Network
===================================================

Combining RWKV ideas with SNN:
- Time-mixing: Weighted average of current and past states
- Channel-mixing: Non-linear transformation with reset gate
- Spiking dynamics: LIF neurons with hybrid readout

Goal: Better long-range memory while maintaining SNN efficiency

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count


class RWKVInspiredSNN:
    """
    RWKV-Inspired SNN with gating mechanisms:
    - Time decay (like RWKV's time mixing)
    - Reset gate (like GRU)
    - Spike + Membrane hybrid readout
    """
    
    def __init__(self, vocab_size, hidden_size=300, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        
        # Time-mixing weights (RWKV style)
        self.time_decay = np.random.uniform(0.5, 0.99, hidden_size)  # Learnable decay per neuron
        self.time_first = np.random.uniform(0.1, 0.5, hidden_size)   # First-step bonus
        
        # Channel-mixing weights
        self.W_key = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_value = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_receptance = np.random.randn(hidden_size, hidden_size) * 0.1  # Reset gate
        
        # Reservoir (sparse)
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        
        # Output
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.lr = 0.1
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)  # Membrane potential
        spike_counts = np.zeros(self.hidden_size)
        
        # State for time-mixing (RWKV style)
        state = np.zeros(self.hidden_size)
        
        W_res_masked = self.W_res * self.mask
        
        for idx, char_idx in enumerate(sequence):
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            # Input current
            I_in = x @ self.W_in
            
            # Time-mixing (RWKV style: weighted average of current and state)
            if idx == 0:
                mixed = I_in * self.time_first + state * (1 - self.time_first)
            else:
                mixed = I_in * (1 - self.time_decay) + state * self.time_decay
            
            # Channel-mixing (RWKV style: key, value, receptance)
            key = mixed @ self.W_key
            value = mixed @ self.W_value
            receptance = self.sigmoid(mixed @ self.W_receptance)  # Reset gate
            
            # Apply receptance (gate)
            channel_out = receptance * (key * value / (np.abs(key).max() + 1e-10))
            
            # Update state
            state = mixed
            
            # LIF dynamics with gated input
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_masked @ spiking * 0.3
                
                v = v * 0.9 + channel_out * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        # Hybrid readout
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        output = features @ self.W_out
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, features
    
    def train_step(self, sequence, target):
        probs, features = self.forward(sequence)
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W_out += self.lr * np.outer(features, target_vec - probs)
        return -np.log(probs[target] + 1e-10)


class StandardSNN:
    """Standard SNN for comparison"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.lr = 0.1
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        W_res_masked = self.W_res * self.mask
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in * 2.0
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_masked @ spiking * 0.3
                v = v * 0.9 + I_in * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        output = features @ self.W_out
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        return probs, features
    
    def train_step(self, sequence, target):
        probs, features = self.forward(sequence)
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W_out += self.lr * np.outer(features, target_vec - probs)
        return -np.log(probs[target] + 1e-10)


def prepare_data(text, seq_length=30):  # Longer sequences to test memory
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    sequences, targets = [], []
    for i in range(0, len(text) - seq_length - 1, seq_length // 2):
        seq = [char_to_idx.get(c, 0) for c in text[i:i+seq_length]]
        tgt = char_to_idx.get(text[i+seq_length], 0)
        sequences.append(seq)
        targets.append(tgt)
    return np.array(sequences), np.array(targets), vocab_size


def get_text():
    # Longer patterns to test memory
    text = """
    once upon a time there was a small village nestled in the mountains
    the villagers lived simple lives farming and trading with neighbors
    every spring the river would swell with melting snow from the peaks
    the children would play by the riverside catching fish and frogs
    as summer approached the fields would turn golden with ripe wheat
    the harvest festival was the most important event of the year
    everyone would gather in the town square to celebrate and dance
    winter brought quiet times with families gathered around warm fires
    stories of ancient heroes and magical creatures filled the nights
    the elders would teach the young ones the wisdom of their ancestors
    """ * 30
    return text.lower()


def train_worker(args):
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'rwkv':
        model = RWKVInspiredSNN(vocab_size, hidden_size, seed)
    else:
        model = StandardSNN(vocab_size, hidden_size, seed)
    
    for _ in range(epochs):
        for i in range(0, len(train_seq), 2):
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    print("=" * 70)
    print("   RWKV-INSPIRED SNN: GATED RECURRENT SPIKING NETWORK")
    print("   Testing long-range memory with RWKV-style time mixing")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    print(f"  Sequence length: 30 (long!)")
    
    n_models = min(8, cpu_count())
    epochs = 12
    
    # RWKV-Inspired (300n)
    print(f"\n  Training RWKV-Inspired SNN (300n)...")
    rwkv_args = [(
        'rwkv', vocab_size, 300, train_seq, train_tgt, 42 + i, epochs
    ) for i in range(n_models)]
    
    t0 = time.time()
    with Pool(n_models) as pool:
        rwkv_models = pool.map(train_worker, rwkv_args)
    rwkv_time = time.time() - t0
    
    # Standard (200n)
    print("  Training Standard SNN (200n)...")
    std_args = [(
        'standard', vocab_size, 200, train_seq, train_tgt, 42 + i, epochs
    ) for i in range(n_models)]
    
    t0 = time.time()
    with Pool(n_models) as pool:
        std_models = pool.map(train_worker, std_args)
    std_time = time.time() - t0
    
    # Test
    print("  Testing...")
    
    def test_models(models):
        losses = []
        for model in models:
            for i in range(len(test_seq)):
                probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
        return np.exp(np.mean(losses))
    
    rwkv_ppl = test_models(rwkv_models)
    std_ppl = test_models(std_models)
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS: RWKV-INSPIRED SNN vs STANDARD")
    print("=" * 70)
    
    rwkv_vs_std = (rwkv_ppl - std_ppl) / std_ppl * 100
    
    print(f"""
    ┌─────────────────────────────┬────────────┬────────────┐
    │ Model                       │ PPL        │ Gap        │
    ├─────────────────────────────┼────────────┼────────────┤
    │ RWKV-Inspired SNN (300n)    │ {rwkv_ppl:10.2f} │ {rwkv_vs_std:+10.1f}% │
    │ Standard SNN (200n)         │ {std_ppl:10.2f} │ baseline   │
    └─────────────────────────────┴────────────┴────────────┘
    
    Training time: RWKV {rwkv_time:.1f}s, Standard {std_time:.1f}s
    """)
    
    if rwkv_ppl < std_ppl:
        print("  🎉 RWKV-INSPIRED BEATS STANDARD!")
        print(f"     Improvement: {-rwkv_vs_std:.1f}%")
    
    print(f"""
    RWKV FEATURES:
    ──────────────
    • Time-mixing: Weighted average of current and past states
    • Channel-mixing: Key, Value, Receptance (like Attention)
    • Gating: Reset gate controls information flow
    • Spiking: LIF dynamics preserved for efficiency
    """)
    
    # Save
    with open("results/rwkv_snn_results.txt", "w", encoding="utf-8") as f:
        f.write("RWKV-Inspired SNN Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"RWKV-Inspired (300n): PPL={rwkv_ppl:.2f}\n")
        f.write(f"Standard (200n): PPL={std_ppl:.2f}\n")
        f.write(f"Gap: {rwkv_vs_std:+.1f}%\n")
    
    print("\n  Results saved to: results/rwkv_snn_results.txt")
    
    return rwkv_ppl, std_ppl


if __name__ == "__main__":
    main()
