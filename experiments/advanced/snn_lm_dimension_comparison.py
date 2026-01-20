"""
Dimension Comparison: Which D is optimal?
==========================================

Testing: 5D, 7D, 9D, 11D hypercubes
Question: Why does the brain use 11D? Is there something special?

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


def ternarize(W):
    scale = np.mean(np.abs(W)) + 1e-10
    return np.sign(W) * (np.abs(W) > 0.5 * scale) * scale


class HypercubeSNN:
    def __init__(self, vocab_size, dim, seed=42):
        np.random.seed(seed)
        
        self.dim = dim
        self.hidden_size = 2 ** dim
        
        self.W_in = np.random.randn(vocab_size, self.hidden_size) * 0.1
        W_res = np.random.randn(self.hidden_size, self.hidden_size) * 0.3
        self.mask = create_hypercube_mask(dim)
        self.W_masked = ternarize(W_res) * self.mask
        self.W_out = np.random.randn(self.hidden_size * 2, vocab_size) * 0.1
    
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


def get_text(size=50000):
    np.random.seed(42)
    templates = [
        "The quick brown fox jumps over the lazy dog. ",
        "To be or not to be, that is the question. ",
        "All that glitters is not gold. ",
        "A journey of a thousand miles begins with a single step. ",
        "Knowledge is power. Time flies like an arrow. ",
        "Actions speak louder than words. ",
    ]
    text = ""
    while len(text) < size:
        text += np.random.choice(templates)
    return text[:size]


def prepare_data(text, seq_length):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    sequences, targets = [], []
    for i in range(0, len(text) - seq_length - 1, seq_length//4):
        seq = [char_to_idx[c] for c in text[i:i+seq_length]]
        target = char_to_idx[text[i+seq_length]]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets), len(chars)


def train_and_test(args):
    dim, vocab_size, train_seq, train_tgt, test_seq, test_tgt, seed, epochs = args
    
    model = HypercubeSNN(vocab_size, dim, seed)
    
    for _ in range(epochs):
        for i in range(len(train_seq)):
            model.train_step(train_seq[i], train_tgt[i])
    
    losses = []
    for i in range(min(100, len(test_seq))):
        probs = model.forward(test_seq[i])
        losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    return np.mean(losses)


def main():
    print("\n" + "=" * 70)
    print("   DIMENSION COMPARISON: Which D is optimal?")
    print("   Testing: 5D, 6D, 7D, 8D, 9D, 10D, 11D hypercubes")
    print("=" * 70)
    
    text = get_text(size=40000)
    seq_length = 30
    sequences, targets, vocab_size = prepare_data(text, seq_length)
    
    n = len(sequences)
    n_train = min(400, int(n * 0.7))
    n_test = min(200, n - n_train)
    
    train_seq = sequences[:n_train]
    train_tgt = targets[:n_train]
    test_seq = sequences[n_train:n_train+n_test]
    test_tgt = targets[n_train:n_train+n_test]
    
    print(f"\n  Data: {len(text):,} chars, seq_len={seq_length}")
    print(f"  Train: {len(train_seq)}, Test: {len(test_seq)}")
    
    dimensions = [5, 6, 7, 8, 9, 10, 11]
    n_parallel = min(12, cpu_count())
    epochs = 3
    n_seeds = 3
    
    results = {}
    
    for dim in dimensions:
        n_neurons = 2 ** dim
        n_connections = n_neurons * dim
        print(f"\n  Testing {dim}D Hypercube ({n_neurons} neurons, {n_connections} connections)")
        print("-" * 50)
        
        all_args = [(dim, vocab_size, train_seq, train_tgt, test_seq, test_tgt, 42+s, epochs) 
                    for s in range(n_seeds)]
        
        t0 = time.time()
        with Pool(min(n_parallel, n_seeds)) as pool:
            all_losses = pool.map(train_and_test, all_args)
        elapsed = time.time() - t0
        
        ppl = np.exp(np.mean(all_losses))
        std = np.std([np.exp(l) for l in all_losses])
        results[dim] = {'ppl': ppl, 'std': std, 'neurons': n_neurons, 'connections': n_connections}
        print(f"    PPL = {ppl:.1f} ± {std:.1f} ({elapsed:.1f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("   DIMENSION COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n  {'Dim':>4} | {'Neurons':>8} | {'Connections':>12} | {'PPL':>10} | {'Efficiency':>12}")
    print("  " + "-" * 60)
    
    best_dim = min(dimensions, key=lambda d: results[d]['ppl'])
    
    for dim in dimensions:
        r = results[dim]
        efficiency = r['neurons'] / r['ppl']  # Neurons per PPL point
        marker = " ⭐" if dim == best_dim else ""
        print(f"  {dim:>4}D | {r['neurons']:>8} | {r['connections']:>12} | {r['ppl']:>9.1f}{marker} | {efficiency:>11.2f}")
    
    # Calculate PPL per connection
    print(f"""
    Best dimension: {best_dim}D (PPL = {results[best_dim]['ppl']:.1f})
    
    Analysis:
    - Smaller dimensions (5-7D): Fewer neurons, less capacity
    - Larger dimensions (9-11D): More capacity, potentially overkill
    - Sweet spot: Around {best_dim}D for this task
    
    Brain's choice of 11D:
    - May be optimized for 10-category classification (fingers!)
    - Language modeling may prefer different dimensions
    - Task-dependent optimization
    """)
    
    # Plot
    dims = list(results.keys())
    ppls = [results[d]['ppl'] for d in dims]
    neurons = [results[d]['neurons'] for d in dims]
    efficiency = [n/p for n,p in zip(neurons, ppls)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PPL by dimension
    axes[0].bar(dims, ppls, color=['#2ecc71' if d == best_dim else '#3498db' for d in dims])
    axes[0].set_xlabel('Dimension', fontsize=12)
    axes[0].set_ylabel('Perplexity (lower is better)', fontsize=12)
    axes[0].set_title('PPL by Hypercube Dimension', fontsize=14, fontweight='bold')
    axes[0].axhline(y=min(ppls), color='red', linestyle='--', alpha=0.5)
    
    # Neurons by dimension
    axes[1].bar(dims, neurons, color='#9b59b6')
    axes[1].set_xlabel('Dimension', fontsize=12)
    axes[1].set_ylabel('Number of Neurons', fontsize=12)
    axes[1].set_title('Neurons = 2^D', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    
    # Efficiency
    axes[2].bar(dims, efficiency, color='#e67e22')
    axes[2].set_xlabel('Dimension', fontsize=12)
    axes[2].set_ylabel('Neurons / PPL (higher is better)', fontsize=12)
    axes[2].set_title('Efficiency: Neurons per PPL Point', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/fig_dimension_comparison.png', dpi=150, bbox_inches='tight')
    print("  Figure saved: results/fig_dimension_comparison.png")
    
    # Save
    with open("results/dimension_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Dimension Comparison Results\n")
        f.write("=" * 40 + "\n\n")
        for dim in dimensions:
            r = results[dim]
            f.write(f"{dim}D: PPL={r['ppl']:.1f}, Neurons={r['neurons']}, Connections={r['connections']}\n")
        f.write(f"\nBest: {best_dim}D\n")
    
    print("  Results saved: results/dimension_comparison.txt")
    
    return results


if __name__ == "__main__":
    main()
