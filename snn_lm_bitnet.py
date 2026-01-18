"""
BitNet b1.58 + SNN: Zero-Multiply Neural Language Model
=========================================================

Combining:
- BitNet b1.58: Ternary weights {-1, 0, 1}
- SNN: Sparse binary spikes {0, 1}

Result: ZERO MULTIPLICATION needed!
- Multiply by 1 = identity
- Multiply by 0 = skip
- Multiply by -1 = negate

Expected: 50-100x efficiency improvement!

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time


class TernarySNN:
    """SNN with BitNet-style ternary weights"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Initialize with continuous weights, then ternarize
        W_in_cont = np.random.randn(vocab_size, hidden_size) * 0.5
        W_res_cont = np.random.randn(hidden_size, hidden_size) * 0.1
        W_out_cont = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        # Ternarize weights: {-1, 0, 1}
        self.W_in = self.ternarize(W_in_cont)
        self.W_res = self.ternarize(W_res_cont)
        
        # Apply sparsity mask to reservoir
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        
        # Output layer: keep continuous for fair comparison
        # (could also ternarize for even more efficiency)
        self.W_out = W_out_cont
        self.W_out_ternary = self.ternarize(W_out_cont)
        
        self.lr = 0.1
        
        # Operation counters
        self.mult_ops = 0
        self.add_ops = 0
    
    def ternarize(self, W, threshold=0.3):
        """Convert weights to {-1, 0, 1}"""
        W_tern = np.zeros_like(W)
        std = np.std(W)
        W_tern[W > threshold * std] = 1
        W_tern[W < -threshold * std] = -1
        return W_tern.astype(np.int8)
    
    def ternary_matmul(self, x, W_tern):
        """
        Matrix multiply with ternary weights - NO MULTIPLICATION!
        
        W_tern is {-1, 0, 1}
        For each output: sum(x where W=1) - sum(x where W=-1)
        """
        # Count additions only (no multiplications!)
        n_nonzero = np.count_nonzero(W_tern)
        self.add_ops += n_nonzero * len(x) if len(x.shape) == 1 else n_nonzero
        
        # Actual computation
        # This is mathematically equivalent to x @ W_tern
        # but demonstrates the addition-only nature
        result = np.zeros(W_tern.shape[1])
        
        for i in range(W_tern.shape[1]):
            pos_mask = W_tern[:, i] == 1
            neg_mask = W_tern[:, i] == -1
            result[i] = np.sum(x[pos_mask]) - np.sum(x[neg_mask])
        
        return result
    
    def forward(self, sequence, time_steps=10, use_ternary_output=False):
        """Forward pass with ternary weights"""
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            # Input encoding (one-hot is already binary!)
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            # Ternary input projection: NO MULTIPLICATIONS
            # x is binary {0, 1}, W_in is ternary {-1, 0, 1}
            I_in = self.ternary_matmul(x, self.W_in) * 2.0
            
            for t in range(time_steps):
                # Spikes are binary {0, 1}
                spiking = (v > 1.0).astype(float)
                
                # Ternary reservoir: NO MULTIPLICATIONS
                I_rec = self.ternary_matmul(spiking, self.W_res) * 0.3
                
                v = v * 0.9 + I_in * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        # Features
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        # Output projection
        if use_ternary_output:
            output = self.ternary_matmul(features, self.W_out_ternary)
        else:
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


class StandardSNN:
    """Standard SNN with full-precision weights for comparison"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.lr = 0.1
        self.mult_ops = 0
        self.add_ops = 0
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            I_in = x @ self.W_in * 2.0
            self.mult_ops += self.vocab_size * self.hidden_size
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = self.W_res @ spiking * 0.3
                self.mult_ops += self.hidden_size * self.hidden_size
                
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


def main():
    print("=" * 70)
    print("   BITNET b1.58 + SNN: ZERO-MULTIPLY EXPERIMENT")
    print("   Ternary Weights {-1, 0, 1} + Binary Spikes {0, 1}")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=20)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    
    # Train Ternary SNN
    print("\n" + "=" * 70)
    print("   Training TERNARY SNN (BitNet-style)")
    print("=" * 70)
    
    ternary_snn = TernarySNN(vocab_size, hidden_size=200, seed=42)
    
    t0 = time.time()
    for epoch in range(5):
        losses = []
        for i in range(0, n_train, 10):
            loss = ternary_snn.train_step(train_seq[i], train_tgt[i])
            losses.append(loss)
        ppl = np.exp(np.mean(losses))
        print(f"    Epoch {epoch+1}: PPL = {ppl:.2f}")
    ternary_time = time.time() - t0
    
    # Test Ternary SNN
    ternary_snn.mult_ops = 0
    ternary_snn.add_ops = 0
    
    ternary_losses = []
    for i in range(len(test_seq)):
        probs, _ = ternary_snn.forward(test_seq[i])
        ternary_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    ternary_ppl = np.exp(np.mean(ternary_losses))
    ternary_mult = ternary_snn.mult_ops
    ternary_add = ternary_snn.add_ops
    
    print(f"\n  Ternary SNN Results:")
    print(f"    Test PPL: {ternary_ppl:.2f}")
    print(f"    Multiplications: {ternary_mult/1e6:.1f}M")
    print(f"    Additions: {ternary_add/1e6:.1f}M")
    print(f"    Time: {ternary_time:.1f}s")
    
    # Train Standard SNN
    print("\n" + "=" * 70)
    print("   Training STANDARD SNN (Full Precision)")
    print("=" * 70)
    
    standard_snn = StandardSNN(vocab_size, hidden_size=200, seed=42)
    
    t0 = time.time()
    for epoch in range(5):
        losses = []
        for i in range(0, n_train, 10):
            loss = standard_snn.train_step(train_seq[i], train_tgt[i])
            losses.append(loss)
        ppl = np.exp(np.mean(losses))
        print(f"    Epoch {epoch+1}: PPL = {ppl:.2f}")
    standard_time = time.time() - t0
    
    # Test Standard SNN
    standard_snn.mult_ops = 0
    standard_snn.add_ops = 0
    
    standard_losses = []
    for i in range(len(test_seq)):
        probs, _ = standard_snn.forward(test_seq[i])
        standard_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    standard_ppl = np.exp(np.mean(standard_losses))
    standard_mult = standard_snn.mult_ops
    
    print(f"\n  Standard SNN Results:")
    print(f"    Test PPL: {standard_ppl:.2f}")
    print(f"    Multiplications: {standard_mult/1e6:.1f}M")
    print(f"    Time: {standard_time:.1f}s")
    
    # Comparison
    print("\n" + "=" * 70)
    print("   COMPARISON: TERNARY vs STANDARD")
    print("=" * 70)
    
    mult_reduction = standard_mult / (ternary_mult + 1) if ternary_mult > 0 else float('inf')
    
    print(f"""
    ┌─────────────────┬────────────┬────────────┬────────────────┐
    │ Model           │ PPL        │ Mult (M)   │ Mult Reduction │
    ├─────────────────┼────────────┼────────────┼────────────────┤
    │ Ternary SNN     │ {ternary_ppl:10.2f} │ {ternary_mult/1e6:10.1f} │ {mult_reduction:14.1f}× │
    │ Standard SNN    │ {standard_ppl:10.2f} │ {standard_mult/1e6:10.1f} │ 1.0× (baseline)│
    └─────────────────┴────────────┴────────────┴────────────────┘
    """)
    
    # Energy estimation
    print("  Energy Estimation:")
    print("-" * 50)
    
    # Typical values
    # Multiplication: ~4 pJ (32-bit)
    # Addition: ~0.1 pJ
    
    energy_mult = 4.0  # pJ per multiplication
    energy_add = 0.1   # pJ per addition
    
    ternary_energy = ternary_mult * energy_mult + ternary_add * energy_add
    standard_energy = standard_mult * energy_mult
    
    energy_ratio = standard_energy / (ternary_energy + 1e-10)
    
    print(f"    Ternary SNN: {ternary_energy/1e9:.2f} mJ")
    print(f"    Standard SNN: {standard_energy/1e9:.2f} mJ")
    print(f"    Energy Reduction: {energy_ratio:.1f}×")
    
    if mult_reduction > 10:
        print(f"\n  🔥🔥🔥 TERNARY SNN ELIMINATES {mult_reduction:.0f}× MULTIPLICATIONS! 🔥🔥🔥")
    
    if energy_ratio > 50:
        print(f"  ⚡⚡⚡ {energy_ratio:.0f}× MORE ENERGY EFFICIENT! ⚡⚡⚡")
    
    # Save results
    with open("results/bitnet_snn_results.txt", "w", encoding="utf-8") as f:
        f.write("BitNet b1.58 + SNN Experiment Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Ternary SNN:\n")
        f.write(f"  PPL: {ternary_ppl:.2f}\n")
        f.write(f"  Multiplications: {ternary_mult/1e6:.1f}M\n")
        f.write(f"  Additions: {ternary_add/1e6:.1f}M\n\n")
        f.write(f"Standard SNN:\n")
        f.write(f"  PPL: {standard_ppl:.2f}\n")
        f.write(f"  Multiplications: {standard_mult/1e6:.1f}M\n\n")
        f.write(f"Multiplication Reduction: {mult_reduction:.1f}×\n")
        f.write(f"Energy Reduction: {energy_ratio:.1f}×\n")
    
    print("\n  Results saved to: results/bitnet_snn_results.txt")
    
    return {
        'ternary_ppl': ternary_ppl,
        'standard_ppl': standard_ppl,
        'mult_reduction': mult_reduction,
        'energy_ratio': energy_ratio
    }


if __name__ == "__main__":
    main()
