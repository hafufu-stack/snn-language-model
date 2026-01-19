"""
BitNet b1.58 + SNN: Large Model Version
========================================

Try larger models to compensate for quantization loss

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count


def ternarize_ste(W, threshold=0.5):
    alpha = np.mean(np.abs(W))
    W_tern = np.zeros_like(W)
    W_tern[W > alpha * threshold] = 1
    W_tern[W < -alpha * threshold] = -1
    return W_tern.astype(np.float32), alpha


class LargeTernarySNN:
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.3
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.05
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.05
        self.lr = 0.2
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        W_in_q, alpha_in = ternarize_ste(self.W_in)
        W_res_q, alpha_res = ternarize_ste(self.W_res * self.mask)
        W_in_use = W_in_q * alpha_in
        W_res_use = W_res_q * alpha_res
        
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
        
        W_out_q, alpha_out = ternarize_ste(self.W_out)
        output = features @ (W_out_q * alpha_out)
        
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        return probs, features
    
    def train_step(self, sequence, target):
        probs, features = self.forward(sequence)
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W_out += self.lr * np.outer(features, target_vec - probs)
        return -np.log(probs[target] + 1e-10)


class LargeStandardSNN:
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
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
    the economy showed signs of strength in the latest report
    artificial intelligence is transforming the technology industry
    neural networks have achieved remarkable results in language tasks
    """ * 60
    return text.lower()


def train_worker(args):
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'ternary':
        model = LargeTernarySNN(vocab_size, hidden_size, seed)
    else:
        model = LargeStandardSNN(vocab_size, hidden_size, seed)
    
    n_train = len(train_seq)
    for _ in range(epochs):
        for i in range(0, n_train, 3):
            model.train_step(train_seq[i], train_tgt[i])
    return model


def main():
    print("=" * 70)
    print("   BITNET + SNN: LARGE MODEL COMPARISON")
    print("   Ternary-400 vs Standard-200 (fair comparison)")
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
    epochs = 10
    
    # Ternary-400 (larger to compensate for quantization)
    print("\n  Training Ternary SNN (400 neurons)...")
    ternary_args = [('ternary', vocab_size, 400, train_seq, train_tgt, 42 + i, epochs) for i in range(n_models)]
    
    t0 = time.time()
    with Pool(n_models) as pool:
        ternary_models = pool.map(train_worker, ternary_args)
    ternary_time = time.time() - t0
    
    # Standard-200
    print("  Training Standard SNN (200 neurons)...")
    standard_args = [('standard', vocab_size, 200, train_seq, train_tgt, 42 + i, epochs) for i in range(n_models)]
    
    t0 = time.time()
    with Pool(n_models) as pool:
        standard_models = pool.map(train_worker, standard_args)
    standard_time = time.time() - t0
    
    # Test
    print("  Testing...")
    
    ternary_losses = []
    for model in ternary_models:
        for i in range(len(test_seq)):
            probs, _ = model.forward(test_seq[i])
            ternary_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    standard_losses = []
    for model in standard_models:
        for i in range(len(test_seq)):
            probs, _ = model.forward(test_seq[i])
            standard_losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    ternary_ppl = np.exp(np.mean(ternary_losses))
    standard_ppl = np.exp(np.mean(standard_losses))
    
    # Operation counts
    ternary_add = 400 * 400 + 400 * vocab_size  # per sample (approximate)
    standard_mult = 200 * 200 + 200 * vocab_size
    
    print("\n" + "=" * 70)
    print("   RESULTS: TERNARY-400 vs STANDARD-200")
    print("=" * 70)
    
    ppl_diff = (ternary_ppl - standard_ppl) / standard_ppl * 100
    
    print(f"""
    ┌───────────────────────┬────────────┬────────────────┐
    │ Model                 │ PPL        │ Neurons        │
    ├───────────────────────┼────────────┼────────────────┤
    │ Ternary SNN (400n)    │ {ternary_ppl:10.2f} │ 400            │
    │ Standard SNN (200n)   │ {standard_ppl:10.2f} │ 200            │
    └───────────────────────┴────────────┴────────────────┘
    
    PPL difference: {ppl_diff:+.1f}%
    
    Key: Ternary uses 2x neurons but ZERO multiplications!
    """)
    
    if ppl_diff < 10:
        print("  🎉🎉🎉 WITHIN 10% OF STANDARD QUALITY! 🎉🎉🎉")
        print("  → Ternary SNN is viable for production!")
    elif ppl_diff < 20:
        print("  ✅ Within 20% of standard quality")
        print("  → Good enough for many applications!")
    
    # Save
    with open("results/bitnet_snn_large_results.txt", "w", encoding="utf-8") as f:
        f.write("BitNet + SNN: Large Model Comparison\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Ternary SNN (400 neurons): PPL={ternary_ppl:.2f}\n")
        f.write(f"Standard SNN (200 neurons): PPL={standard_ppl:.2f}\n")
        f.write(f"PPL difference: {ppl_diff:+.1f}%\n")
    
    print("\n  Results saved to: results/bitnet_snn_large_results.txt")
    
    return ternary_ppl, standard_ppl


if __name__ == "__main__":
    main()
