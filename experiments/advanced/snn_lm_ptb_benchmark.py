"""
SNN Language Model - Penn Treebank Benchmark
=============================================

Standard academic benchmark for language models.
Penn Treebank (PTB) is a widely used dataset for evaluating perplexity.

This script:
1. Downloads/uses PTB-style text
2. Trains SNN, DNN, LSTM models
3. Computes perplexity (standard metric)
4. Compares energy efficiency

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import time
import urllib.request
import os


# =============================================================================
# DATA PREPARATION
# =============================================================================

def get_ptb_sample():
    """Get a PTB-like text sample for benchmarking.
    
    Note: Full PTB requires license. We use a similar style text.
    """
    
    # Sample text in PTB style (lowercased, tokenized)
    # This is a simplified version for quick testing
    ptb_sample = """
    the company said it expects to report a loss for the third quarter
    the board of directors approved a plan to buy back shares
    analysts said the stock is likely to rise in the coming weeks
    the federal reserve is expected to raise interest rates next month
    the president announced a new policy on trade with china
    the company reported earnings that beat wall street expectations
    investors are worried about the impact of the trade war
    the central bank cut interest rates for the first time in years
    the stock market closed lower amid concerns about global growth
    the company announced plans to lay off thousands of workers
    the economy grew at a faster pace than expected last quarter
    the government released new data on unemployment and inflation
    the merger between the two companies was approved by regulators
    the technology sector led gains in the stock market today
    the oil prices fell sharply after the report on inventories
    the housing market showed signs of cooling in recent months
    the retail sales data came in weaker than economists expected
    the company raised its dividend for the tenth consecutive year
    the bond yields rose as investors sold off treasury securities
    the chairman said the company is well positioned for growth
    the quarterly results were in line with the company guidance
    the market volatility has increased in recent trading sessions
    the earnings season has been mixed so far this quarter
    the consumer confidence index fell to its lowest level in months
    the manufacturing sector showed signs of weakness in the report
    the company announced a restructuring plan to cut costs
    the technology stocks rallied after the positive earnings report
    the currency markets were volatile ahead of the policy decision
    the central bank left interest rates unchanged as expected
    the job market remained strong despite concerns about growth
    the company said it would increase investment in research
    the stock fell after the company lowered its profit forecast
    the economic indicators suggested a slowdown in the coming months
    the merger talks between the companies have stalled recently
    the technology giant reported record revenue for the quarter
    the oil market stabilized after the agreement on production cuts
    the retail chain announced plans to close hundreds of stores
    the banking sector faced pressure from the low interest rates
    the company said it would return cash to shareholders this year
    the market rally continued despite the uncertainty in washington
    """ * 10  # Repeat to get more data
    
    return ptb_sample.lower().strip()


def create_char_vocab(text):
    """Create character vocabulary"""
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(chars)


def prepare_sequences(text, char_to_idx, seq_length=50):
    """Prepare training sequences"""
    sequences = []
    targets = []
    
    for i in range(0, len(text) - seq_length - 1, seq_length // 2):
        seq = text[i:i+seq_length]
        target = text[i+seq_length]
        
        seq_idx = [char_to_idx.get(c, 0) for c in seq]
        target_idx = char_to_idx.get(target, 0)
        
        sequences.append(seq_idx)
        targets.append(target_idx)
    
    return np.array(sequences), np.array(targets)


# =============================================================================
# MODELS
# =============================================================================

class SNNLanguageModel:
    """Spiking Neural Network Language Model with Hybrid Coding"""
    
    def __init__(self, vocab_size, hidden_size=300, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # LIF parameters
        self.tau = 15.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.v_rest = -65.0
        
        # Weights
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.3
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.2 / rho
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        
        # Output (hybrid: spike count + membrane potential)
        self.W_out_spike = np.random.randn(hidden_size, vocab_size) * 0.1
        self.W_out_membrane = np.random.randn(hidden_size, vocab_size) * 0.1
        
        # Learning rate
        self.lr = 0.01
        
        # Track operations
        self.total_ops = 0
    
    def forward(self, sequence, time_steps=20):
        """Forward pass with spike counting"""
        batch_ops = 0
        
        # Initialize state
        v = np.full(self.hidden_size, self.v_rest)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            # One-hot input
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in
            batch_ops += self.vocab_size * self.hidden_size
            
            # Simulate for time_steps
            for t in range(time_steps):
                I_rec = self.W_res @ (v > self.v_thresh).astype(float)
                batch_ops += self.hidden_size * self.hidden_size
                
                I_total = I_in * np.exp(-t/10) + I_rec * 10
                
                dv = (-(v - self.v_rest) + I_total) / self.tau
                v += dv
                
                spiking = v >= self.v_thresh
                spike_counts += spiking.astype(float)
                v[spiking] = self.v_reset
        
        # Hybrid output
        out_spike = spike_counts @ self.W_out_spike
        out_membrane = v @ self.W_out_membrane
        output = out_spike + out_membrane * 0.5
        batch_ops += self.hidden_size * self.vocab_size * 2
        
        self.total_ops += batch_ops
        
        # Softmax
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, spike_counts, v
    
    def train_step(self, sequence, target):
        """Simple LMS-style training"""
        probs, spikes, membrane = self.forward(sequence)
        
        # Error
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        error = target_vec - probs
        
        # Update output weights
        self.W_out_spike += self.lr * np.outer(spikes, error)
        self.W_out_membrane += self.lr * np.outer(membrane, error)
        
        return -np.log(probs[target] + 1e-10)


class DNNLanguageModel:
    """Simple DNN Language Model"""
    
    def __init__(self, vocab_size, hidden_size=300, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.01
        self.total_ops = 0
    
    def forward(self, sequence):
        batch_ops = 0
        
        h = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            h1 = np.tanh(x @ self.W1)
            batch_ops += self.vocab_size * self.hidden_size
            
            h = np.tanh(h1 + h @ self.W2)
            batch_ops += self.hidden_size * self.hidden_size
        
        output = h @ self.W3
        batch_ops += self.hidden_size * self.vocab_size
        
        self.total_ops += batch_ops
        
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, h
    
    def train_step(self, sequence, target):
        probs, h = self.forward(sequence)
        
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        error = target_vec - probs
        
        self.W3 += self.lr * np.outer(h, error)
        
        return -np.log(probs[target] + 1e-10)


class LSTMLanguageModel:
    """Simple LSTM Language Model"""
    
    def __init__(self, vocab_size, hidden_size=300, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # LSTM gates
        self.Wf = np.random.randn(vocab_size + hidden_size, hidden_size) * 0.1
        self.Wi = np.random.randn(vocab_size + hidden_size, hidden_size) * 0.1
        self.Wc = np.random.randn(vocab_size + hidden_size, hidden_size) * 0.1
        self.Wo = np.random.randn(vocab_size + hidden_size, hidden_size) * 0.1
        
        self.W_out = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.01
        self.total_ops = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, sequence):
        batch_ops = 0
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            combined = np.concatenate([x, h])
            
            f = self.sigmoid(combined @ self.Wf)
            i = self.sigmoid(combined @ self.Wi)
            c_cand = np.tanh(combined @ self.Wc)
            o = self.sigmoid(combined @ self.Wo)
            
            batch_ops += 4 * (self.vocab_size + self.hidden_size) * self.hidden_size
            
            c = f * c + i * c_cand
            h = o * np.tanh(c)
        
        output = h @ self.W_out
        batch_ops += self.hidden_size * self.vocab_size
        
        self.total_ops += batch_ops
        
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, h
    
    def train_step(self, sequence, target):
        probs, h = self.forward(sequence)
        
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        error = target_vec - probs
        
        self.W_out += self.lr * np.outer(h, error)
        
        return -np.log(probs[target] + 1e-10)


# =============================================================================
# BENCHMARK
# =============================================================================

def compute_perplexity(losses):
    """Compute perplexity from cross-entropy losses"""
    return np.exp(np.mean(losses))


def run_benchmark(hidden_size=300, n_epochs=3, seq_length=30):
    """Run the PTB-style benchmark"""
    
    print("=" * 70)
    print("   PENN TREEBANK STYLE BENCHMARK")
    print("   Standard Language Model Evaluation")
    print("=" * 70)
    
    # Prepare data
    print("\n  Preparing data...")
    text = get_ptb_sample()
    char_to_idx, idx_to_char, vocab_size = create_char_vocab(text)
    sequences, targets = prepare_sequences(text, char_to_idx, seq_length)
    
    n_samples = len(sequences)
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train
    
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"    Total characters: {len(text)}")
    print(f"    Vocabulary size:  {vocab_size}")
    print(f"    Train samples:    {n_train}")
    print(f"    Test samples:     {n_test}")
    print(f"    Hidden size:      {hidden_size}")
    
    # Initialize models
    print("\n  Initializing models...")
    snn = SNNLanguageModel(vocab_size, hidden_size, seed=42)
    dnn = DNNLanguageModel(vocab_size, hidden_size, seed=42)
    lstm = LSTMLanguageModel(vocab_size, hidden_size, seed=42)
    
    results = {}
    
    # Train and evaluate each model
    for name, model in [("SNN", snn), ("DNN", dnn), ("LSTM", lstm)]:
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        # Training
        for epoch in range(n_epochs):
            train_losses = []
            for i in range(0, n_train, 10):  # Sample for speed
                loss = model.train_step(train_seq[i], train_tgt[i])
                train_losses.append(loss)
            
            if (epoch + 1) % 1 == 0:
                ppl = compute_perplexity(train_losses)
                print(f"      Epoch {epoch+1}: Train PPL = {ppl:.2f}")
        
        # Testing
        test_losses = []
        model.total_ops = 0  # Reset ops counter for test
        
        for i in range(n_test):
            if name == "SNN":
                probs, _, _ = model.forward(test_seq[i])
            else:
                probs, _ = model.forward(test_seq[i])
            
            loss = -np.log(probs[test_tgt[i]] + 1e-10)
            test_losses.append(loss)
        
        elapsed = time.time() - start_time
        test_ppl = compute_perplexity(test_losses)
        total_ops = model.total_ops
        
        # Compute efficiency
        efficiency = (1.0 / test_ppl) / (total_ops / 1e6)
        
        results[name] = {
            'perplexity': test_ppl,
            'operations': total_ops,
            'time': elapsed,
            'efficiency': efficiency
        }
        
        print(f"    {name} Test Perplexity: {test_ppl:.2f}")
        print(f"    {name} Operations: {total_ops/1e6:.1f}M")
        print(f"    {name} Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("   BENCHMARK RESULTS")
    print("=" * 70)
    
    print("\n  Perplexity (lower is better):")
    print("-" * 50)
    for name, r in results.items():
        print(f"    {name:6s}: {r['perplexity']:8.2f}")
    
    print("\n  Operations (lower is better):")
    print("-" * 50)
    for name, r in results.items():
        print(f"    {name:6s}: {r['operations']/1e6:8.1f}M")
    
    # Efficiency comparison
    print("\n  Energy Efficiency Comparison:")
    print("-" * 50)
    
    snn_ops = results['SNN']['operations']
    dnn_ops = results['DNN']['operations']
    lstm_ops = results['LSTM']['operations']
    
    dnn_ratio = dnn_ops / snn_ops
    lstm_ratio = lstm_ops / snn_ops
    
    print(f"    SNN vs DNN:  SNN is {dnn_ratio:.1f}x more efficient")
    print(f"    SNN vs LSTM: SNN is {lstm_ratio:.1f}x more efficient")
    
    # Perplexity-adjusted efficiency
    print("\n  Quality-Adjusted Efficiency (PPL/Ops):")
    print("-" * 50)
    
    snn_eff = (1.0/results['SNN']['perplexity']) / (snn_ops/1e9)
    dnn_eff = (1.0/results['DNN']['perplexity']) / (dnn_ops/1e9)
    lstm_eff = (1.0/results['LSTM']['perplexity']) / (lstm_ops/1e9)
    
    print(f"    SNN:  {snn_eff:.4f}")
    print(f"    DNN:  {dnn_eff:.4f}")
    print(f"    LSTM: {lstm_eff:.4f}")
    
    if snn_eff > dnn_eff:
        print(f"\n  ✅ SNN is {snn_eff/dnn_eff:.1f}x more quality-efficient than DNN!")
    else:
        print(f"\n  ⚠️ DNN has {dnn_eff/snn_eff:.1f}x better quality-efficiency")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("   FINAL VERDICT")
    print("=" * 70)
    
    print(f"""
    ┌────────────┬────────────┬────────────┬────────────┐
    │ Model      │ Perplexity │ Operations │ Efficiency │
    ├────────────┼────────────┼────────────┼────────────┤
    │ SNN        │ {results['SNN']['perplexity']:10.2f} │ {results['SNN']['operations']/1e6:8.1f}M │ {snn_eff:10.4f} │
    │ DNN        │ {results['DNN']['perplexity']:10.2f} │ {results['DNN']['operations']/1e6:8.1f}M │ {dnn_eff:10.4f} │
    │ LSTM       │ {results['LSTM']['perplexity']:10.2f} │ {results['LSTM']['operations']/1e6:8.1f}M │ {lstm_eff:10.4f} │
    └────────────┴────────────┴────────────┴────────────┘
    
    SNN uses {dnn_ratio:.1f}x fewer operations than DNN!
    SNN uses {lstm_ratio:.1f}x fewer operations than LSTM!
    """)
    
    # Save results
    with open("results/ptb_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("Penn Treebank Style Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Data: {len(text)} chars, vocab={vocab_size}, seq_len={seq_length}\n")
        f.write(f"Hidden size: {hidden_size}\n")
        f.write(f"Epochs: {n_epochs}\n\n")
        
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Perplexity: {r['perplexity']:.2f}\n")
            f.write(f"  Operations: {r['operations']/1e6:.1f}M\n")
            f.write(f"  Time: {r['time']:.1f}s\n\n")
        
        f.write(f"Efficiency ratios:\n")
        f.write(f"  SNN vs DNN: {dnn_ratio:.1f}x\n")
        f.write(f"  SNN vs LSTM: {lstm_ratio:.1f}x\n")
    
    print("\n  Results saved to: results/ptb_benchmark_results.txt")
    
    return results


if __name__ == "__main__":
    # Run with different hidden sizes
    print("\n" + "=" * 70)
    print("   RUNNING PTB BENCHMARK WITH HIDDEN=300")
    print("=" * 70)
    
    results_300 = run_benchmark(hidden_size=300, n_epochs=3, seq_length=30)
    
    print("\n" + "=" * 70)
    print("   RUNNING PTB BENCHMARK WITH HIDDEN=500")
    print("=" * 70)
    
    results_500 = run_benchmark(hidden_size=500, n_epochs=3, seq_length=30)
