"""
BitNet b1.58 + SNN: Zero-Multiply Neural Language Model (Parallel Version)
===========================================================================

Combining:
- BitNet b1.58: Ternary weights {-1, 0, 1}
- SNN: Sparse binary spikes {0, 1}
- Multiprocessing for faster execution

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def ternarize(W, threshold=0.3):
    """Convert weights to {-1, 0, 1}"""
    W_tern = np.zeros_like(W)
    std = np.std(W)
    W_tern[W > threshold * std] = 1
    W_tern[W < -threshold * std] = -1
    return W_tern.astype(np.int8)


def ternary_matmul_fast(x, W_tern):
    """Fast ternary matmul using numpy"""
    # This is equivalent to x @ W_tern but shows the logic
    # For production: just use x @ W_tern (numpy handles it efficiently)
    return x @ W_tern.astype(np.float32)


class TernarySNN:
    """SNN with BitNet-style ternary weights"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Initialize and ternarize
        W_in_cont = np.random.randn(vocab_size, hidden_size) * 0.5
        W_res_cont = np.random.randn(hidden_size, hidden_size) * 0.1
        
        self.W_in = ternarize(W_in_cont)
        self.W_res = ternarize(W_res_cont)
        
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.1
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = ternary_matmul_fast(x, self.W_in) * 2.0
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = ternary_matmul_fast(spiking, self.W_res) * 0.3
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


class StandardSNN:
    """Standard SNN with full-precision weights"""
    
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
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in * 2.0
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = self.W_res @ spiking * 0.3
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
    """ * 40
    return text.lower()


def train_model_worker(args):
    """Worker function for parallel training"""
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed = args
    
    if model_type == 'ternary':
        model = TernarySNN(vocab_size, hidden_size, seed)
    else:
        model = StandardSNN(vocab_size, hidden_size, seed)
    
    n_train = len(train_seq)
    for _ in range(5):
        for i in range(0, n_train, 10):
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def test_model_worker(args):
    """Worker function for parallel testing"""
    model, test_seq, test_tgt = args
    
    losses = []
    for i in range(len(test_seq)):
        probs, _ = model.forward(test_seq[i])
        losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
    
    return np.mean(losses)


def run_parallel_experiment(vocab_size, hidden_size, train_seq, train_tgt, test_seq, test_tgt, n_models=4):
    """Run multiple models in parallel"""
    
    print(f"\n  Running {n_models} models in parallel (using {cpu_count()} CPUs)...")
    
    # Prepare arguments for parallel training
    ternary_args = [('ternary', vocab_size, hidden_size, train_seq, train_tgt, 42 + i) for i in range(n_models)]
    standard_args = [('standard', vocab_size, hidden_size, train_seq, train_tgt, 42 + i) for i in range(n_models)]
    
    # Train in parallel
    print("  Training Ternary SNNs...")
    t0 = time.time()
    with Pool(min(n_models, cpu_count())) as pool:
        ternary_models = pool.map(train_model_worker, ternary_args)
    ternary_train_time = time.time() - t0
    
    print("  Training Standard SNNs...")
    t0 = time.time()
    with Pool(min(n_models, cpu_count())) as pool:
        standard_models = pool.map(train_model_worker, standard_args)
    standard_train_time = time.time() - t0
    
    # Test sequentially (models have state)
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
    
    return {
        'ternary_ppl': ternary_ppl,
        'standard_ppl': standard_ppl,
        'ternary_time': ternary_train_time,
        'standard_time': standard_train_time,
        'n_models': n_models
    }


def main():
    print("=" * 70)
    print("   BITNET b1.58 + SNN: PARALLEL VERSION")
    print("   Using multiprocessing for faster experimentation")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=20)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    print(f"  CPUs available: {cpu_count()}")
    
    # Run parallel experiment
    results = run_parallel_experiment(
        vocab_size, 200, 
        train_seq, train_tgt, 
        test_seq, test_tgt,
        n_models=min(8, cpu_count())
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────┬────────────┬────────────┐
    │ Model           │ PPL        │ Train Time │
    ├─────────────────┼────────────┼────────────┤
    │ Ternary SNN     │ {results['ternary_ppl']:10.2f} │ {results['ternary_time']:10.1f}s │
    │ Standard SNN    │ {results['standard_ppl']:10.2f} │ {results['standard_time']:10.1f}s │
    └─────────────────┴────────────┴────────────┘
    
    Models trained in parallel: {results['n_models']}
    """)
    
    # Energy analysis
    print("  Theoretical Energy Analysis:")
    print("-" * 50)
    print("  Ternary weights {-1, 0, 1} → multiply = shift/negate/zero")
    print("  Standard weights → full FP32 multiplication")
    print("  Theoretical reduction: ~10-100x depending on hardware")
    
    # Save results
    with open("results/bitnet_snn_parallel_results.txt", "w", encoding="utf-8") as f:
        f.write("BitNet b1.58 + SNN Parallel Experiment\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Ternary SNN PPL: {results['ternary_ppl']:.2f}\n")
        f.write(f"Standard SNN PPL: {results['standard_ppl']:.2f}\n")
        f.write(f"Models trained: {results['n_models']}\n")
        f.write(f"Ternary train time: {results['ternary_time']:.1f}s\n")
        f.write(f"Standard train time: {results['standard_time']:.1f}s\n")
    
    print("\n  Results saved to: results/bitnet_snn_parallel_results.txt")
    
    return results


if __name__ == "__main__":
    main()
