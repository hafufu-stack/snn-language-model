"""
BitNet b1.58 + SNN: Mixed Precision V3 (Bug Fixed)
===================================================

Fixed issues from V2:
- More stable initialization
- Fixed scale factor (not learnable, caused instability)
- Simpler approach that works

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count


class MixedPrecisionV3SNN:
    """
    Mixed Precision SNN V3 - Stable Version
    Key: Keep it simple, focus on what works
    """
    
    def __init__(self, vocab_size, hidden_size=500, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Standard initialization (proven to work)
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        # Sparsity mask
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        
        self.lr = 0.1
    
    def ternarize(self, W):
        """Simple ternarization without learnable scale"""
        alpha = np.mean(np.abs(W))
        threshold = 0.5
        W_tern = np.zeros_like(W)
        W_tern[W > alpha * threshold] = 1
        W_tern[W < -alpha * threshold] = -1
        return W_tern * alpha
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        # Only ternarize reservoir (mixed precision)
        W_res_masked = self.W_res * self.mask
        W_res_tern = self.ternarize(W_res_masked)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            # Input: CONTINUOUS (full precision)
            I_in = x @ self.W_in * 2.0
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                
                # Reservoir: TERNARY
                I_rec = W_res_tern @ spiking * 0.3
                
                v = v * 0.9 + I_in * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        # Output: CONTINUOUS (full precision)
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
    """Standard full-precision SNN"""
    
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


def prepare_data(text, seq_length=20):
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
    text = """
    the company said it expects to report a loss for the third quarter
    the board of directors approved a plan to buy back shares
    analysts said the stock is likely to rise in the coming weeks
    the federal reserve is expected to raise interest rates next month
    technology companies led gains in the market today overall
    investors are watching for signals from the central bank
    """ * 50
    return text.lower()


def train_worker(args):
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'mixed_v3':
        model = MixedPrecisionV3SNN(vocab_size, hidden_size, seed)
    else:
        model = StandardSNN(vocab_size, hidden_size, seed)
    
    for _ in range(epochs):
        for i in range(0, len(train_seq), 2):
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    print("=" * 70)
    print("   BITNET + SNN: MIXED PRECISION V3 (STABLE)")
    print("   500n Mixed vs 200n Standard")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=20)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    
    n_models = min(8, cpu_count())
    epochs = 12
    
    # Mixed V3 (500n)
    print(f"\n  Training Mixed Precision V3 (500n)...")
    mixed_args = [(
        'mixed_v3', vocab_size, 500, train_seq, train_tgt, 42 + i, epochs
    ) for i in range(n_models)]
    
    t0 = time.time()
    with Pool(n_models) as pool:
        mixed_models = pool.map(train_worker, mixed_args)
    mixed_time = time.time() - t0
    
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
    
    mixed_ppl = test_models(mixed_models)
    std_ppl = test_models(std_models)
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS: MIXED PRECISION V3 vs STANDARD")
    print("=" * 70)
    
    mixed_vs_std = (mixed_ppl - std_ppl) / std_ppl * 100
    
    print(f"""
    ┌─────────────────────────────┬────────────┬────────────┐
    │ Model                       │ PPL        │ Gap        │
    ├─────────────────────────────┼────────────┼────────────┤
    │ Mixed Precision V3 (500n)   │ {mixed_ppl:10.2f} │ {mixed_vs_std:+10.1f}% │
    │ Standard SNN (200n)         │ {std_ppl:10.2f} │ baseline   │
    └─────────────────────────────┴────────────┴────────────┘
    
    Training time: Mixed {mixed_time:.1f}s, Standard {std_time:.1f}s
    """)
    
    if mixed_vs_std < 20:
        print("  🎉🎉🎉 WITHIN 20% OF STANDARD! GOAL ACHIEVED! 🎉🎉🎉")
    elif mixed_vs_std < 30:
        print("  ✅✅ Within 30% - Very good!")
    elif mixed_vs_std < 50:
        print("  ✅ Within 50% - Good progress!")
    else:
        print(f"  Gap is {mixed_vs_std:.1f}% - needs more work")
    
    # Save
    with open("results/bitnet_snn_mixed_v3_results.txt", "w", encoding="utf-8") as f:
        f.write("BitNet + SNN: Mixed Precision V3 Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mixed Precision V3 (500n): PPL={mixed_ppl:.2f}\n")
        f.write(f"Standard (200n): PPL={std_ppl:.2f}\n")
        f.write(f"Gap: {mixed_vs_std:+.1f}%\n")
    
    print("\n  Results saved to: results/bitnet_snn_mixed_v3_results.txt")
    
    return mixed_ppl, std_ppl, mixed_vs_std


if __name__ == "__main__":
    main()
