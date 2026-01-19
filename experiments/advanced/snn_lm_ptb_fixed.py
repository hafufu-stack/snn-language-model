"""
SNN Language Model - Penn Treebank Benchmark (Fixed Version)
=============================================================

Fixed issues:
1. Better output scaling for SNN
2. Improved learning rate
3. Better spike-to-probability conversion

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time


def get_ptb_sample():
    """PTB-style text"""
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
    """ * 20
    return ptb_sample.lower().strip()


class ImprovedSNNLanguageModel:
    """Fixed SNN with better learning"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Simpler LIF
        self.tau = 10.0
        self.v_thresh = 1.0
        self.v_reset = 0.0
        
        # Input embedding
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        
        # Reservoir
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        
        # Output: both spike count and membrane
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.lr = 0.1
        self.total_ops = 0
    
    def forward(self, sequence, time_steps=10):
        """Forward with proper normalization"""
        ops = 0
        
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in * 2.0
            ops += self.vocab_size * self.hidden_size
            
            for t in range(time_steps):
                I_rec = self.W_res @ (v > self.v_thresh).astype(float)
                ops += self.hidden_size * self.hidden_size
                
                v = v * 0.9 + I_in * 0.5 + I_rec * 0.3
                
                spiking = v > self.v_thresh
                spike_counts += spiking.astype(float)
                v[spiking] = self.v_reset
        
        # Normalize features
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        
        features = np.concatenate([spike_norm, v_norm])
        output = features @ self.W_out
        ops += self.hidden_size * 2 * self.vocab_size
        
        self.total_ops += ops
        
        # Softmax with temperature
        output = output / 1.0
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, features
    
    def train_step(self, sequence, target):
        probs, features = self.forward(sequence)
        
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        error = target_vec - probs
        
        self.W_out += self.lr * np.outer(features, error)
        
        return -np.log(probs[target] + 1e-10)


class SimpleDNN:
    """Simple DNN baseline"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.1
        self.total_ops = 0
    
    def forward(self, sequence):
        ops = 0
        h = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            h1 = np.tanh(x @ self.W1)
            ops += self.vocab_size * self.hidden_size
            
            h = np.tanh(h1 * 0.5 + h @ self.W2 * 0.5)
            ops += self.hidden_size * self.hidden_size
        
        output = h @ self.W3
        ops += self.hidden_size * self.vocab_size
        
        self.total_ops += ops
        
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


class SimpleLSTM:
    """Simple LSTM baseline"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        d = vocab_size + hidden_size
        self.Wf = np.random.randn(d, hidden_size) * 0.1
        self.Wi = np.random.randn(d, hidden_size) * 0.1
        self.Wc = np.random.randn(d, hidden_size) * 0.1
        self.Wo = np.random.randn(d, hidden_size) * 0.1
        self.W_out = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.1
        self.total_ops = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, sequence):
        ops = 0
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
            
            ops += 4 * (self.vocab_size + self.hidden_size) * self.hidden_size
            
            c = f * c + i * c_cand
            h = o * np.tanh(c)
        
        output = h @ self.W_out
        ops += self.hidden_size * self.vocab_size
        
        self.total_ops += ops
        
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


def run_fixed_benchmark():
    print("=" * 70)
    print("   PTB BENCHMARK (FIXED VERSION)")
    print("=" * 70)
    
    text = get_ptb_sample()
    sequences, targets, vocab_size = prepare_data(text, seq_length=20)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    
    hidden = 200
    epochs = 5
    
    results = {}
    
    for name, Model in [("SNN", ImprovedSNNLanguageModel), 
                        ("DNN", SimpleDNN), 
                        ("LSTM", SimpleLSTM)]:
        print(f"\n  Training {name}...")
        model = Model(vocab_size, hidden, seed=42)
        
        t0 = time.time()
        
        for epoch in range(epochs):
            losses = []
            for i in range(0, n_train, 5):
                loss = model.train_step(train_seq[i], train_tgt[i])
                losses.append(loss)
            ppl = np.exp(np.mean(losses))
            print(f"    Epoch {epoch+1}: PPL = {ppl:.2f}")
        
        # Test
        model.total_ops = 0
        test_losses = []
        for i in range(len(test_seq)):
            probs, _ = model.forward(test_seq[i])
            loss = -np.log(probs[test_tgt[i]] + 1e-10)
            test_losses.append(loss)
        
        test_ppl = np.exp(np.mean(test_losses))
        total_ops = model.total_ops
        elapsed = time.time() - t0
        
        results[name] = {
            'ppl': test_ppl,
            'ops': total_ops,
            'time': elapsed
        }
        
        print(f"    Test PPL: {test_ppl:.2f}, Ops: {total_ops/1e6:.1f}M")
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS SUMMARY")
    print("=" * 70)
    
    snn_ops = results['SNN']['ops']
    dnn_ops = results['DNN']['ops']
    lstm_ops = results['LSTM']['ops']
    
    snn_ppl = results['SNN']['ppl']
    dnn_ppl = results['DNN']['ppl']
    lstm_ppl = results['LSTM']['ppl']
    
    print(f"""
    ┌────────┬────────────┬───────────┬─────────────────────┐
    │ Model  │ Perplexity │ Ops (M)   │ Efficiency Ratio    │
    ├────────┼────────────┼───────────┼─────────────────────┤
    │ SNN    │ {snn_ppl:10.2f} │ {snn_ops/1e6:9.1f} │ 1.0x (baseline)     │
    │ DNN    │ {dnn_ppl:10.2f} │ {dnn_ops/1e6:9.1f} │ {dnn_ops/snn_ops:.1f}x more ops       │
    │ LSTM   │ {lstm_ppl:10.2f} │ {lstm_ops/1e6:9.1f} │ {lstm_ops/snn_ops:.1f}x more ops       │
    └────────┴────────────┴───────────┴─────────────────────┘
    """)
    
    # Quality-adjusted efficiency
    snn_eff = (1.0/snn_ppl) / (snn_ops/1e9)
    dnn_eff = (1.0/dnn_ppl) / (dnn_ops/1e9)
    lstm_eff = (1.0/lstm_ppl) / (lstm_ops/1e9)
    
    print("  Quality-Adjusted Efficiency (higher = better):")
    print(f"    SNN:  {snn_eff:.6f}")
    print(f"    DNN:  {dnn_eff:.6f}")
    print(f"    LSTM: {lstm_eff:.6f}")
    
    if snn_eff > dnn_eff:
        print(f"\n  ✅ SNN is {snn_eff/dnn_eff:.1f}x more efficient than DNN!")
    else:
        ratio = dnn_eff / snn_eff
        print(f"\n  SNN efficiency ratio vs DNN: {1/ratio:.2f}x")
    
    # Save
    with open("results/ptb_fixed_results.txt", "w", encoding="utf-8") as f:
        f.write("PTB Benchmark (Fixed) Results\n")
        f.write("=" * 40 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}: PPL={r['ppl']:.2f}, Ops={r['ops']/1e6:.1f}M\n")
        f.write(f"\nSNN vs DNN ops: {dnn_ops/snn_ops:.1f}x\n")
        f.write(f"SNN vs LSTM ops: {lstm_ops/snn_ops:.1f}x\n")
    
    print("\n  Results saved to: results/ptb_fixed_results.txt")
    
    return results


if __name__ == "__main__":
    run_fixed_benchmark()
