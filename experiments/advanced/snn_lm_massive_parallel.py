"""
Massive Parallel Ultimate SNN: 16 Models x Maximum CPU
=======================================================

Utilizing full CPU power with 16 parallel models

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count


def ternarize(W):
    alpha = np.mean(np.abs(W))
    W_tern = np.zeros_like(W)
    W_tern[W > alpha * 0.5] = 1
    W_tern[W < -alpha * 0.5] = -1
    return W_tern * alpha


class UltimateMaxSNN:
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        self.time_decay = np.random.uniform(0.7, 0.95, hidden_size)
        self.W_key = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_value = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_gate = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        state = np.zeros(self.hidden_size)
        
        W_res_tern = ternarize(self.W_res * self.mask)
        
        for idx, char_idx in enumerate(sequence):
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            I_in = x @ self.W_in
            mixed = I_in * (1 - self.time_decay) + state * self.time_decay
            
            key = mixed @ self.W_key
            value = mixed @ self.W_value
            gate = self.sigmoid(mixed @ self.W_gate)
            
            channel_out = gate * (key * value / (np.abs(key).max() + 1e-10))
            state = mixed
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + channel_out * 0.5 + I_rec
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


def prepare_data(text, seq_length=30):
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
    once upon a time there was a small village nestled in the mountains
    the villagers lived simple lives farming and trading with neighbors
    every spring the river would swell with melting snow from the peaks
    neural networks learn patterns from vast amounts of training data
    spiking networks mimic the brain using discrete pulses of activity
    membrane potentials encode analog information between spike events
    artificial intelligence is transforming industries around the world
    """ * 40
    return text.lower()


def train_worker(args):
    model_type, vocab_size, hidden_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'ultimate':
        model = UltimateMaxSNN(vocab_size, hidden_size, seed)
    else:
        model = StandardSNN(vocab_size, hidden_size, seed)
    
    for _ in range(epochs):
        for i in range(0, len(train_seq), 2):
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    n_parallel = min(22, cpu_count())  # Use up to 22 cores!
    
    print("=" * 70)
    print(f"   MASSIVE PARALLEL ULTIMATE SNN: {n_parallel} PARALLEL MODELS")
    print("   Utilizing maximum CPU power!")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    print(f"  CPUs available: {cpu_count()}")
    print(f"  Using: {n_parallel} parallel workers")
    
    epochs = 12
    
    # Ultimate (400n) - 16 models!
    print(f"\n  Training {n_parallel} Ultimate SNN models in parallel...")
    ult_args = [(
        'ultimate', vocab_size, 400, train_seq, train_tgt, 42 + i, epochs
    ) for i in range(n_parallel)]
    
    t0 = time.time()
    with Pool(n_parallel) as pool:
        ult_models = pool.map(train_worker, ult_args)
    ult_time = time.time() - t0
    
    # Standard (200n) - 16 models!
    print(f"  Training {n_parallel} Standard SNN models in parallel...")
    std_args = [(
        'standard', vocab_size, 200, train_seq, train_tgt, 42 + i, epochs
    ) for i in range(n_parallel)]
    
    t0 = time.time()
    with Pool(n_parallel) as pool:
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
    
    ult_ppl = test_models(ult_models)
    std_ppl = test_models(std_models)
    
    ult_vs_std = (ult_ppl - std_ppl) / std_ppl * 100
    
    print("\n" + "=" * 70)
    print(f"   RESULTS: {n_parallel} PARALLEL MODELS")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────────┬────────────┬────────────┬──────────┐
    │ Model                       │ PPL        │ Gap        │ Time     │
    ├─────────────────────────────┼────────────┼────────────┼──────────┤
    │ Ultimate SNN (400n x{n_parallel:2d})     │ {ult_ppl:10.2f} │ {ult_vs_std:+10.1f}% │ {ult_time:7.1f}s │
    │ Standard SNN (200n x{n_parallel:2d})     │ {std_ppl:10.2f} │ baseline   │ {std_time:7.1f}s │
    └─────────────────────────────┴────────────┴────────────┴──────────┘
    """)
    
    if ult_ppl < std_ppl:
        print(f"  🎉 ULTIMATE BEATS STANDARD by {-ult_vs_std:.1f}%!")
    
    with open("results/massive_parallel_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Massive Parallel SNN: {n_parallel} Models\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Ultimate (400n x{n_parallel}): PPL={ult_ppl:.2f}, Time={ult_time:.1f}s\n")
        f.write(f"Standard (200n x{n_parallel}): PPL={std_ppl:.2f}, Time={std_time:.1f}s\n")
        f.write(f"Gap: {ult_vs_std:+.1f}%\n")
    
    print("\n  Results saved to: results/massive_parallel_results.txt")


if __name__ == "__main__":
    main()
