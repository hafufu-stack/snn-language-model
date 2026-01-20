"""
Extended Sequence Length Test: 200-500 characters
==================================================

Testing if 11D topology maintains advantage for VERY long sequences.

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for node in range(n):
        for d in range(dim):
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1
    return mask


def create_random_mask(n, sparsity, seed=42):
    np.random.seed(seed)
    mask = (np.random.rand(n, n) < sparsity).astype(float)
    np.fill_diagonal(mask, 0)
    return mask


def ternarize(W):
    scale = np.mean(np.abs(W)) + 1e-10
    return np.sign(W) * (np.abs(W) > 0.5 * scale) * scale


class BaseSNN:
    def __init__(self, vocab_size, hidden_size, mask, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.1
        W_res = np.random.randn(hidden_size, hidden_size) * 0.3
        self.W_masked = ternarize(W_res) * mask
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
    
    def forward(self, sequence, time_steps=10):
        x = np.zeros(self.hidden_size)
        for idx in sequence:
            x += self.W_in[idx]
        x = np.tanh(x)
        
        membrane = x.copy()
        spikes = np.zeros(self.hidden_size)
        for _ in range(time_steps):
            recurrent = membrane @ self.W_masked
            membrane = 0.9 * membrane + 0.1 * recurrent
            fired = (membrane > 1.0).astype(float)
            spikes += fired
            membrane *= (1 - fired)
        
        hybrid = np.concatenate([spikes / time_steps, membrane])
        out = hybrid @ self.W_out
        out = out - np.max(out)
        return np.exp(out) / (np.sum(np.exp(out)) + 1e-10)
    
    def train_step(self, sequence, target):
        probs = self.forward(sequence)
        return -np.log(probs[target] + 1e-10)


def get_text(size=100000):
    np.random.seed(42)
    templates = [
        "The quick brown fox jumps over the lazy dog. ",
        "To be or not to be, that is the question. ",
        "All that glitters is not gold. ",
        "A journey of a thousand miles begins with a single step. ",
        "Knowledge is power. Time flies like an arrow. ",
        "Actions speak louder than words. The pen is mightier than the sword. ",
        "In the beginning was the Word. All roads lead to Rome. ",
        "The only thing we have to fear is fear itself. I think, therefore I am. ",
    ]
    text = ""
    while len(text) < size:
        text += np.random.choice(templates)
    return text[:size]


def prepare_data(text, seq_length):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    sequences, targets = [], []
    step = max(1, seq_length // 3)
    for i in range(0, len(text) - seq_length - 1, step):
        seq = [char_to_idx[c] for c in text[i:i+seq_length]]
        target = char_to_idx[text[i+seq_length]]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets), len(chars)


def train_and_test(args):
    model_type, vocab_size, train_seq, train_tgt, test_seq, test_tgt, seed, epochs = args
    
    hidden_size = 512
    if model_type == 'pure11d':
        mask = create_hypercube_mask(9)
    else:
        mask = create_random_mask(512, sparsity=0.02, seed=seed)
    
    model = BaseSNN(vocab_size, hidden_size, mask, seed)
    
    for _ in range(epochs):
        for i in range(len(train_seq)):
            model.train_step(train_seq[i], train_tgt[i])
    
    losses = []
    for i in range(min(80, len(test_seq))):
        probs = model.forward(test_seq[i])
        losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    return np.mean(losses)


def main():
    print("\n" + "=" * 70)
    print("   EXTENDED SEQUENCE LENGTH TEST")
    print("   Testing: 100, 200, 300, 500 character sequences")
    print("=" * 70)
    
    text = get_text(size=80000)
    seq_lengths = [100, 200, 300, 500]
    model_types = ['pure11d', 'random']
    n_parallel = min(12, cpu_count())
    epochs = 3
    n_seeds = 4
    
    results = {mt: [] for mt in model_types}
    
    for seq_len in seq_lengths:
        print(f"\n  Sequence length: {seq_len}")
        print("-" * 40)
        
        sequences, targets, vocab_size = prepare_data(text, seq_len)
        n = len(sequences)
        n_train = min(300, int(n * 0.6))
        n_test = min(150, n - n_train)
        
        train_seq = sequences[:n_train]
        train_tgt = targets[:n_train]
        test_seq = sequences[n_train:n_train+n_test]
        test_tgt = targets[n_train:n_train+n_test]
        
        all_args = []
        for mt in model_types:
            for seed in range(n_seeds):
                all_args.append((mt, vocab_size, train_seq, train_tgt,
                               test_seq, test_tgt, 42+seed, epochs))
        
        t0 = time.time()
        with Pool(n_parallel) as pool:
            all_losses = pool.map(train_and_test, all_args)
        elapsed = time.time() - t0
        
        idx = 0
        for mt in model_types:
            losses = all_losses[idx:idx+n_seeds]
            ppl = np.exp(np.mean(losses))
            results[mt].append(ppl)
            print(f"    {mt:10s}: PPL = {ppl:.1f}")
            idx += n_seeds
        
        print(f"    (Time: {elapsed:.1f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("   EXTENDED SEQUENCE RESULTS")
    print("=" * 70)
    
    print(f"\n  {'Seq Len':>8} | {'Pure 11D':>10} | {'Random':>10} | {'Advantage':>10} | Winner")
    print("  " + "-" * 55)
    
    for i, seq_len in enumerate(seq_lengths):
        ppl_11d = results['pure11d'][i]
        ppl_rand = results['random'][i]
        advantage = (ppl_rand - ppl_11d) / ppl_rand * 100
        winner = "11D ⭐" if ppl_11d < ppl_rand else "Random"
        print(f"  {seq_len:>8} | {ppl_11d:>10.1f} | {ppl_rand:>10.1f} | {advantage:>+9.1f}% | {winner}")
    
    # Plot extended results
    plt.figure(figsize=(10, 6))
    
    # Add previous results (10-100)
    short_lengths = [10, 20, 30, 50, 100]
    short_11d = [36.4, 39.0, 40.5, 42.3, 44.4]
    short_rand = [37.0, 41.4, 43.5, 45.6, 47.4]
    
    all_lengths = short_lengths + seq_lengths
    all_11d = short_11d + results['pure11d']
    all_rand = short_rand + results['random']
    
    plt.plot(all_lengths, all_11d, 'o-', color='#2ecc71', 
             label='Pure 11D (9D Hypercube)', linewidth=2, markersize=8)
    plt.plot(all_lengths, all_rand, 'o-', color='#e74c3c',
             label='Random Sparse', linewidth=2, markersize=8)
    
    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Original test boundary')
    
    plt.xlabel('Sequence Length (characters)', fontsize=12)
    plt.ylabel('Perplexity (lower is better)', fontsize=12)
    plt.title('11D Topology: Performance vs Sequence Length (Extended)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fig_extended_sequence_length.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: results/fig_extended_sequence_length.png")
    
    # Save
    with open("results/extended_sequence_results.txt", "w", encoding="utf-8") as f:
        f.write("Extended Sequence Length Results\n")
        f.write("=" * 40 + "\n\n")
        for i, seq_len in enumerate(seq_lengths):
            f.write(f"{seq_len}: 11D={results['pure11d'][i]:.1f}, Random={results['random'][i]:.1f}\n")
    
    print("  Results saved: results/extended_sequence_results.txt")
    
    return results


if __name__ == "__main__":
    main()
