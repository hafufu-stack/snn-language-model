"""
Combined Experiments: Progressive + Attention + Ultimate
=========================================================

Combining the winning techniques:
- Progressive Training (best: -28%)
- Attention-SNN (second: -21.5%)
- Ultimate (BitNet + RWKV + Hybrid)

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


# =============================================================================
# Combined Model 1: Progressive + Attention
# =============================================================================
class ProgressiveAttentionSNN:
    """Best of both: Progressive training + Attention mechanism"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.active_size = hidden_size // 4  # Start small
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        
        # Attention weights
        self.W_Q = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_K = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_V = np.random.randn(hidden_size, hidden_size) * 0.1
        
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def grow(self):
        self.active_size = min(self.active_size + self.hidden_size // 4, self.hidden_size)
    
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.active_size)
        spike_counts = np.zeros(self.active_size)
        history = []
        
        W_res_tern = ternarize(self.W_res[:self.active_size, :self.active_size] * 
                               self.mask[:self.active_size, :self.active_size])
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in[:, :self.active_size]
            history.append(I_in.copy())
            
            # Attention over history
            if len(history) > 1:
                H = np.array(history[-5:])
                Q = I_in @ self.W_Q[:self.active_size, :]
                K = H @ self.W_K[:self.active_size, :]
                scores = K @ Q / np.sqrt(self.active_size // 4 + 1)
                attn = self.softmax(scores)
                V = H @ self.W_V[:self.active_size, :self.active_size]
                context = attn @ V
                I_in = I_in + context * 0.3
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + I_in * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        # Pad to full size for output
        spike_full = np.zeros(self.hidden_size)
        spike_full[:self.active_size] = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_full = np.zeros(self.hidden_size)
        v_full[:self.active_size] = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_full, v_full])
        
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


# =============================================================================
# Combined Model 2: Ultimate + Progressive + Attention
# =============================================================================
class SuperUltimateSNN:
    """Everything combined: BitNet + RWKV + Hybrid + Progressive + Attention"""
    
    def __init__(self, vocab_size, hidden_size=500, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.active_size = hidden_size // 4  # Progressive
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        
        # RWKV Time-mixing
        self.time_decay = np.random.uniform(0.7, 0.95, hidden_size)
        self.W_key = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_value = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_gate = np.random.randn(hidden_size, hidden_size) * 0.08
        
        # Attention
        self.W_Q = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_K = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_V = np.random.randn(hidden_size, hidden_size) * 0.1
        
        # BitNet reservoir
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def grow(self):
        self.active_size = min(self.active_size + self.hidden_size // 4, self.hidden_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def forward(self, sequence, time_steps=10):
        h = self.active_size
        v = np.zeros(h)
        spike_counts = np.zeros(h)
        state = np.zeros(h)
        history = []
        
        W_res_tern = ternarize(self.W_res[:h, :h] * self.mask[:h, :h])
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in[:, :h]
            
            # RWKV Time-mixing
            mixed = I_in * (1 - self.time_decay[:h]) + state * self.time_decay[:h]
            
            # RWKV Channel-mixing
            key = mixed @ self.W_key[:h, :h]
            value = mixed @ self.W_value[:h, :h]
            gate = self.sigmoid(mixed @ self.W_gate[:h, :h])
            channel_out = gate * (key * value / (np.abs(key).max() + 1e-10))
            
            state = mixed
            history.append(channel_out.copy())
            
            # Attention enhancement
            if len(history) > 1:
                H = np.array(history[-5:])
                Q = channel_out @ self.W_Q[:h, :]
                K = H @ self.W_K[:h, :]
                scores = K @ Q / np.sqrt(h // 4 + 1)
                attn = self.softmax(scores)
                V = H @ self.W_V[:h, :h]
                context = attn @ V
                channel_out = channel_out + context * 0.2
            
            # Spiking dynamics with BitNet reservoir
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + channel_out * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
        # Pad to full size
        spike_full = np.zeros(self.hidden_size)
        spike_full[:h] = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_full = np.zeros(self.hidden_size)
        v_full[:h] = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_full, v_full])
        
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


# =============================================================================
# Baseline: Standard SNN
# =============================================================================
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


# =============================================================================
# Data and Training
# =============================================================================
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
    neural networks learn patterns from vast amounts of training data
    spiking networks mimic the brain using discrete pulses of activity
    transformers use attention mechanisms to capture long range patterns
    """ * 50
    return text.lower()


def train_progressive_worker(args):
    model_class, vocab_size, train_seq, train_tgt, seed, name = args
    
    if model_class == 'prog_attn':
        model = ProgressiveAttentionSNN(vocab_size, 400, seed)
    elif model_class == 'super_ultimate':
        model = SuperUltimateSNN(vocab_size, 500, seed)
    else:
        model = StandardSNN(vocab_size, 200, seed)
        # Standard training
        for _ in range(12):
            for i in range(0, len(train_seq), 2):
                model.train_step(train_seq[i], train_tgt[i])
        return model, name
    
    # Progressive training: grow 4 times
    for phase in range(4):
        for _ in range(3):  # 3 epochs per phase
            for i in range(0, len(train_seq), 2):
                model.train_step(train_seq[i], train_tgt[i])
        if hasattr(model, 'grow'):
            model.grow()
    
    return model, name


def main():
    n_parallel = min(22, cpu_count())
    
    print("=" * 70)
    print("   COMBINED EXPERIMENTS: Progressive + Attention + Ultimate")
    print(f"   Using {n_parallel} parallel workers")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text)} chars, vocab={vocab_size}")
    print(f"  Train: {n_train}, Test: {n - n_train}")
    
    # Prepare args for parallel training
    models_per_type = n_parallel // 3
    all_args = []
    
    for i in range(models_per_type):
        all_args.append(('prog_attn', vocab_size, train_seq, train_tgt, 42 + i, 'Progressive+Attention'))
        all_args.append(('super_ultimate', vocab_size, train_seq, train_tgt, 100 + i, 'Super Ultimate'))
        all_args.append(('standard', vocab_size, train_seq, train_tgt, 200 + i, 'Standard'))
    
    print(f"\n  Training {len(all_args)} models...")
    t0 = time.time()
    with Pool(n_parallel) as pool:
        results = pool.map(train_progressive_worker, all_args)
    train_time = time.time() - t0
    
    # Group by name
    grouped = {}
    for model, name in results:
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(model)
    
    # Test each group
    print("\n  Testing...")
    
    def test_models(models):
        losses = []
        for model in models:
            for i in range(len(test_seq)):
                probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
        return np.exp(np.mean(losses))
    
    ppl_results = {}
    for name in grouped:
        ppl_results[name] = test_models(grouped[name])
    
    std_ppl = ppl_results.get('Standard', 1.0)
    
    print("\n" + "=" * 70)
    print("   RESULTS: COMBINED EXPERIMENTS")
    print("=" * 70)
    
    print(f"""
    ┌────────────────────────────┬────────────┬────────────┐
    │ Model                      │ PPL        │ vs Standard│
    ├────────────────────────────┼────────────┼────────────┤""")
    
    for name in sorted(ppl_results.keys()):
        ppl = ppl_results[name]
        gap = (ppl - std_ppl) / std_ppl * 100
        marker = "✅" if ppl < std_ppl else "  "
        print(f"    │ {name:26s} │ {ppl:10.2f} │ {gap:+10.1f}% │ {marker}")
    
    print(f"""    └────────────────────────────┴────────────┴────────────┘
    
    Training time: {train_time:.1f}s
    """)
    
    # Find best
    best_name = min(ppl_results, key=ppl_results.get)
    best_ppl = ppl_results[best_name]
    print(f"  🏆 BEST: {best_name} with PPL={best_ppl:.2f}")
    
    improvement = (std_ppl - best_ppl) / std_ppl * 100
    print(f"  📈 IMPROVEMENT: {improvement:.1f}% over Standard")
    
    # Save results
    with open("results/combined_experiments_results.txt", "w", encoding="utf-8") as f:
        f.write("Combined Experiments: Progressive + Attention + Ultimate\n")
        f.write("=" * 50 + "\n\n")
        for name in sorted(ppl_results.keys()):
            ppl = ppl_results[name]
            gap = (ppl - std_ppl) / std_ppl * 100
            f.write(f"{name}: PPL={ppl:.2f}, gap={gap:+.1f}%\n")
        f.write(f"\nBest: {best_name} with PPL={best_ppl:.2f}\n")
        f.write(f"Improvement: {improvement:.1f}% over Standard\n")
    
    print("\n  Results saved to: results/combined_experiments_results.txt")
    
    return ppl_results


if __name__ == "__main__":
    main()
