"""
BitNet b1.58 + SNN: Improved Version with STE
==============================================

Key improvement:
- Train with continuous weights (full precision gradient)
- Quantize to ternary only during FORWARD pass
- Straight-Through Estimator: gradient flows through quantization

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count


def ternarize_ste(W, threshold=0.5):
    """
    Ternarize with Straight-Through Estimator (STE)
    Returns both ternary weights and the scale factor
    """
    # Compute threshold based on mean absolute value (like BitNet b1.58)
    alpha = np.mean(np.abs(W))
    
    # Ternarize
    W_tern = np.zeros_like(W)
    W_tern[W > alpha * threshold] = 1
    W_tern[W < -alpha * threshold] = -1
    
    return W_tern.astype(np.float32), alpha


class ImprovedTernarySNN:
    """
    SNN with improved ternary quantization:
    - Continuous weights stored for training
    - Ternarized on-the-fly during forward pass
    - Gradients flow through via STE
    """
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Keep CONTINUOUS weights for training
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        # Sparsity mask for reservoir
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        
        self.lr = 0.15  # Slightly higher LR for ternary
        
        # Stats
        self.mult_ops = 0
        self.add_ops = 0
    
    def forward(self, sequence, time_steps=10, use_ternary=True):
        """
        Forward pass with optional ternarization
        - use_ternary=True: use ternary weights (for inference)
        - use_ternary=False: use continuous weights (for gradient check)
        """
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        # Get weights (ternary or continuous)
        if use_ternary:
            W_in_q, alpha_in = ternarize_ste(self.W_in)
            W_res_q, alpha_res = ternarize_ste(self.W_res * self.mask)
            
            # Scale by alpha to maintain magnitude
            W_in_use = W_in_q * alpha_in
            W_res_use = W_res_q * alpha_res
            
            # Count ops (ternary = additions only for ternary part)
            self.add_ops += np.count_nonzero(W_in_q) * len(sequence)
        else:
            W_in_use = self.W_in
            W_res_use = self.W_res * self.mask
            self.mult_ops += self.vocab_size * self.hidden_size * len(sequence)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ W_in_use * 2.0
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_use @ spiking * 0.3
                v = v * 0.9 + I_in * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        # Output layer: also ternarize
        if use_ternary:
            W_out_q, alpha_out = ternarize_ste(self.W_out)
            output = features @ (W_out_q * alpha_out)
            self.add_ops += np.count_nonzero(W_out_q)
        else:
            output = features @ self.W_out
            self.mult_ops += len(features) * self.vocab_size
        
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, features
    
    def train_step(self, sequence, target):
        """
        Training with STE:
        - Forward uses ternary weights
        - Gradient updates continuous weights
        """
        # Forward with ternary
        probs, features = self.forward(sequence, use_ternary=True)
        
        # Compute gradient (same as before)
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        
        # Update CONTINUOUS weights (STE: gradient flows through)
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
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.lr = 0.1
        self.mult_ops = 0
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        W_res_masked = self.W_res * self.mask
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in * 2.0
            self.mult_ops += self.vocab_size * self.hidden_size
            
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
        self.mult_ops += len(features) * self.vocab_size
        
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
    the economy showed signs of strength in the latest report
    artificial intelligence is transforming the technology industry
    neural networks have achieved remarkable results in language tasks
    """ * 40
    return text.lower()


def train_worker(args):
    """Worker for parallel training"""
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'improved_ternary':
        model = ImprovedTernarySNN(vocab_size, hidden_size, seed)
    else:
        model = StandardSNN(vocab_size, hidden_size, seed)
    
    n_train = len(train_seq)
    for _ in range(epochs):
        for i in range(0, n_train, 5):  # More frequent updates
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    print("=" * 70)
    print("   BITNET b1.58 + SNN: IMPROVED VERSION (STE)")
    print("   Train with continuous, quantize during forward")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=20)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    print(f"  CPUs: {cpu_count()}")
    
    n_models = min(8, cpu_count())
    epochs = 10  # More epochs
    
    # Prepare args
    improved_args = [('improved_ternary', vocab_size, 200, train_seq, train_tgt, 42 + i, epochs) for i in range(n_models)]
    standard_args = [('standard', vocab_size, 200, train_seq, train_tgt, 42 + i, epochs) for i in range(n_models)]
    
    print(f"\n  Training {n_models} models in parallel, {epochs} epochs each...")
    
    # Train Improved Ternary
    print("\n  Training IMPROVED Ternary SNNs (STE method)...")
    t0 = time.time()
    with Pool(n_models) as pool:
        improved_models = pool.map(train_worker, improved_args)
    improved_time = time.time() - t0
    
    # Train Standard
    print("  Training Standard SNNs...")
    t0 = time.time()
    with Pool(n_models) as pool:
        standard_models = pool.map(train_worker, standard_args)
    standard_time = time.time() - t0
    
    # Test
    print("  Testing...")
    
    improved_losses = []
    improved_add_ops = 0
    improved_mult_ops = 0
    for model in improved_models:
        model.add_ops = 0
        model.mult_ops = 0
        for i in range(len(test_seq)):
            probs, _ = model.forward(test_seq[i], use_ternary=True)
            improved_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
        improved_add_ops += model.add_ops
        improved_mult_ops += model.mult_ops
    
    standard_losses = []
    standard_mult_ops = 0
    for model in standard_models:
        model.mult_ops = 0
        for i in range(len(test_seq)):
            probs, _ = model.forward(test_seq[i])
            standard_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
        standard_mult_ops += model.mult_ops
    
    improved_ppl = np.exp(np.mean(improved_losses))
    standard_ppl = np.exp(np.mean(standard_losses))
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS: IMPROVED vs STANDARD")
    print("=" * 70)
    
    mult_reduction = standard_mult_ops / (improved_mult_ops + 1) if improved_mult_ops > 0 else float('inf')
    
    print(f"""
    ┌─────────────────────┬────────────┬────────────┬────────────────┐
    │ Model               │ PPL        │ Time       │ Mult Ops       │
    ├─────────────────────┼────────────┼────────────┼────────────────┤
    │ Improved Ternary    │ {improved_ppl:10.2f} │ {improved_time:10.1f}s │ {improved_mult_ops/1e6:14.1f}M │
    │ Standard SNN        │ {standard_ppl:10.2f} │ {standard_time:10.1f}s │ {standard_mult_ops/1e6:14.1f}M │
    └─────────────────────┴────────────┴────────────┴────────────────┘
    
    Addition ops (Ternary): {improved_add_ops/1e6:.1f}M
    Multiplication reduction: {mult_reduction:.0f}×
    """)
    
    # Check improvement
    ppl_diff = (improved_ppl - standard_ppl) / standard_ppl * 100
    
    if improved_ppl < 200:
        print(f"  ✅ Improved! PPL = {improved_ppl:.2f}")
        if ppl_diff < 20:
            print(f"  🎉 Within 20% of standard! ({ppl_diff:+.1f}%)")
    
    # Save
    with open("results/bitnet_snn_improved_results.txt", "w", encoding="utf-8") as f:
        f.write("BitNet b1.58 + SNN: Improved STE Method\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Improved Ternary SNN:\n")
        f.write(f"  PPL: {improved_ppl:.2f}\n")
        f.write(f"  Mult ops: {improved_mult_ops/1e6:.1f}M\n")
        f.write(f"  Add ops: {improved_add_ops/1e6:.1f}M\n")
        f.write(f"  Time: {improved_time:.1f}s\n\n")
        f.write(f"Standard SNN:\n")
        f.write(f"  PPL: {standard_ppl:.2f}\n")
        f.write(f"  Mult ops: {standard_mult_ops/1e6:.1f}M\n")
        f.write(f"  Time: {standard_time:.1f}s\n\n")
        f.write(f"Mult reduction: {mult_reduction:.0f}×\n")
        f.write(f"PPL difference: {ppl_diff:+.1f}%\n")
    
    print("\n  Results saved to: results/bitnet_snn_improved_results.txt")
    
    return {
        'improved_ppl': improved_ppl,
        'standard_ppl': standard_ppl,
        'mult_reduction': mult_reduction,
        'ppl_diff': ppl_diff
    }


if __name__ == "__main__":
    main()
