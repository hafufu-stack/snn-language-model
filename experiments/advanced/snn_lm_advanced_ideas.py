"""
Advanced SNN Experiments: 5 Improvement Ideas
==============================================

1. Attention-SNN Hybrid
2. Learnable Time Constants
3. Multi-scale Reservoir
4. Spike Timing Dependent
5. Progressive Training

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
# 1. Attention-SNN Hybrid
# =============================================================================
class AttentionSNN:
    """SNN with lightweight self-attention mechanism"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        
        # Attention weights
        self.W_Q = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_K = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_V = np.random.randn(hidden_size, hidden_size) * 0.1
        
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        history = []
        
        W_res_tern = ternarize(self.W_res * self.mask)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in
            history.append(I_in.copy())
            
            # Simple attention over recent history
            if len(history) > 1:
                H = np.array(history[-5:])  # Last 5 states
                Q = I_in @ self.W_Q
                K = H @ self.W_K
                scores = K @ Q / np.sqrt(self.hidden_size // 4)
                attn = self.softmax(scores)
                V = H @ self.W_V
                context = attn @ V
                I_in = I_in + context * 0.3
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
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
# 2. Learnable Time Constants
# =============================================================================
class LearnableTauSNN:
    """SNN with learnable time constants per neuron"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        # Learnable tau (decay) per neuron
        self.tau = np.random.uniform(0.7, 0.95, hidden_size)
        
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        W_res_tern = ternarize(self.W_res * self.mask)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                # Per-neuron time constant
                v = self.tau * v + (1 - self.tau) * I_in * 0.5 + I_rec
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
# 3. Multi-scale Reservoir
# =============================================================================
class MultiScaleSNN:
    """SNN with multiple reservoirs at different sparsity levels"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.h1 = hidden_size // 3
        self.h2 = hidden_size // 3
        self.h3 = hidden_size - 2 * (hidden_size // 3)
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        
        # Three reservoirs with different sparsity
        self.W_res1 = np.random.randn(self.h1, self.h1) * 0.1
        self.mask1 = np.random.rand(self.h1, self.h1) < 0.05  # Very sparse
        
        self.W_res2 = np.random.randn(self.h2, self.h2) * 0.1
        self.mask2 = np.random.rand(self.h2, self.h2) < 0.1   # Medium
        
        self.W_res3 = np.random.randn(self.h3, self.h3) * 0.1
        self.mask3 = np.random.rand(self.h3, self.h3) < 0.2   # Dense
        
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def forward(self, sequence, time_steps=10):
        v1 = np.zeros(self.h1)
        v2 = np.zeros(self.h2)
        v3 = np.zeros(self.h3)
        s1, s2, s3 = np.zeros(self.h1), np.zeros(self.h2), np.zeros(self.h3)
        
        W1 = ternarize(self.W_res1 * self.mask1)
        W2 = ternarize(self.W_res2 * self.mask2)
        W3 = ternarize(self.W_res3 * self.mask3)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in
            
            I1, I2, I3 = I_in[:self.h1], I_in[self.h1:self.h1+self.h2], I_in[self.h1+self.h2:]
            
            for t in range(time_steps):
                sp1 = (v1 > 1.0).astype(float)
                sp2 = (v2 > 1.0).astype(float)
                sp3 = (v3 > 1.0).astype(float)
                
                v1 = v1 * 0.9 + I1 * 0.5 + W1 @ sp1 * 0.3
                v2 = v2 * 0.9 + I2 * 0.5 + W2 @ sp2 * 0.3
                v3 = v3 * 0.9 + I3 * 0.5 + W3 @ sp3 * 0.3
                
                s1 += sp1
                s2 += sp2
                s3 += sp3
                
                v1[sp1 > 0] = 0
                v2[sp2 > 0] = 0
                v3[sp3 > 0] = 0
        
        norm = len(sequence) * time_steps + 1e-10
        spikes = np.concatenate([s1/norm, s2/norm, s3/norm])
        memb = np.concatenate([v1, v2, v3])
        memb = memb / (np.abs(memb).max() + 1e-10)
        features = np.concatenate([spikes, memb])
        
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
# 4. Spike Timing Dependent
# =============================================================================
class SpikeTimingSNN:
    """SNN using spike timing information"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        # 3 features per neuron: count, first spike time, last spike time
        self.W_out = np.random.randn(hidden_size * 4, vocab_size) * 0.1
        self.lr = 0.12
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        first_spike = np.full(self.hidden_size, -1.0)
        last_spike = np.full(self.hidden_size, -1.0)
        
        W_res_tern = ternarize(self.W_res * self.mask)
        total_steps = len(sequence) * time_steps
        step = 0
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + I_in * 0.5 + I_rec
                
                # Track timing
                spike_mask = spiking > 0
                first_spike = np.where((first_spike < 0) & spike_mask, step / total_steps, first_spike)
                last_spike = np.where(spike_mask, step / total_steps, last_spike)
                
                spike_counts += spiking
                v[spiking > 0] = 0
                step += 1
        
        spike_norm = spike_counts / (total_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        first_norm = np.where(first_spike < 0, 0, first_spike)
        last_norm = np.where(last_spike < 0, 0, last_spike)
        
        features = np.concatenate([spike_norm, v_norm, first_norm, last_norm])
        
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
# 5. Progressive Training (small -> large)
# =============================================================================
class ProgressiveSNN:
    """SNN with progressive training: start small, grow larger"""
    
    def __init__(self, vocab_size, hidden_size=400, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.active_size = hidden_size // 4  # Start with 1/4
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
    
    def grow(self):
        """Increase active neurons"""
        self.active_size = min(self.active_size + self.hidden_size // 4, self.hidden_size)
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        W_res_tern = ternarize(self.W_res[:self.active_size, :self.active_size] * 
                               self.mask[:self.active_size, :self.active_size])
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in[:, :self.active_size]
            
            for t in range(time_steps):
                spiking = (v[:self.active_size] > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v[:self.active_size] = v[:self.active_size] * 0.9 + I_in * 0.5 + I_rec
                spike_counts[:self.active_size] += spiking
                v[:self.active_size][spiking > 0] = 0
        
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
# Baseline Standard SNN
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
    """ * 30
    return text.lower()


def train_worker(args):
    model_type, vocab_size, train_seq, train_tgt, seed, epochs = args
    
    if model_type == 'attention':
        model = AttentionSNN(vocab_size, 400, seed)
    elif model_type == 'learnable_tau':
        model = LearnableTauSNN(vocab_size, 400, seed)
    elif model_type == 'multiscale':
        model = MultiScaleSNN(vocab_size, 400, seed)
    elif model_type == 'spike_timing':
        model = SpikeTimingSNN(vocab_size, 400, seed)
    elif model_type == 'progressive':
        model = ProgressiveSNN(vocab_size, 400, seed)
        for phase in range(4):
            for i in range(0, len(train_seq), 2):
                model.train_step(train_seq[i], train_tgt[i])
            model.grow()
        return model
    else:
        model = StandardSNN(vocab_size, 200, seed)
    
    for _ in range(epochs):
        for i in range(0, len(train_seq), 2):
            model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    n_parallel = min(22, cpu_count())
    
    print("=" * 70)
    print("   ADVANCED SNN EXPERIMENTS: 5 IMPROVEMENT IDEAS")
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
    
    epochs = 12
    models_per_type = n_parallel // 6  # Split across 6 model types
    if models_per_type < 1:
        models_per_type = 1
    
    model_types = ['attention', 'learnable_tau', 'multiscale', 
                   'spike_timing', 'progressive', 'standard']
    
    all_args = []
    for mtype in model_types:
        for i in range(models_per_type):
            all_args.append((mtype, vocab_size, train_seq, train_tgt, 42 + i, epochs))
    
    print(f"\n  Training {len(all_args)} models...")
    t0 = time.time()
    with Pool(n_parallel) as pool:
        all_models = pool.map(train_worker, all_args)
    train_time = time.time() - t0
    
    # Group models by type
    results = {}
    for mtype in model_types:
        results[mtype] = []
    
    for i, model in enumerate(all_models):
        mtype = all_args[i][0]
        results[mtype].append(model)
    
    # Test each type
    print("\n  Testing...")
    
    def test_models(models):
        losses = []
        for model in models:
            for i in range(len(test_seq)):
                probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
        return np.exp(np.mean(losses))
    
    ppl_results = {}
    for mtype in model_types:
        ppl_results[mtype] = test_models(results[mtype])
    
    std_ppl = ppl_results['standard']
    
    print("\n" + "=" * 70)
    print("   RESULTS: 5 IMPROVEMENT IDEAS")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────┬────────────┬────────────┐
    │ Model                   │ PPL        │ vs Standard│
    ├─────────────────────────┼────────────┼────────────┤""")
    
    for mtype in model_types:
        ppl = ppl_results[mtype]
        gap = (ppl - std_ppl) / std_ppl * 100
        name = mtype.replace('_', ' ').title()[:23]
        marker = "✅" if ppl < std_ppl else "  "
        print(f"    │ {name:23s} │ {ppl:10.2f} │ {gap:+10.1f}% │ {marker}")
    
    print(f"""    └─────────────────────────┴────────────┴────────────┘
    
    Training time: {train_time:.1f}s
    """)
    
    # Find best
    best_type = min(ppl_results, key=ppl_results.get)
    best_ppl = ppl_results[best_type]
    print(f"  🏆 BEST: {best_type.replace('_', ' ').title()} with PPL={best_ppl:.2f}")
    
    # Save results
    with open("results/advanced_experiments_results.txt", "w", encoding="utf-8") as f:
        f.write("Advanced SNN Experiments: 5 Improvement Ideas\n")
        f.write("=" * 50 + "\n\n")
        for mtype in model_types:
            ppl = ppl_results[mtype]
            gap = (ppl - std_ppl) / std_ppl * 100
            f.write(f"{mtype}: PPL={ppl:.2f}, gap={gap:+.1f}%\n")
        f.write(f"\nBest: {best_type} with PPL={best_ppl:.2f}\n")
    
    print("\n  Results saved to: results/advanced_experiments_results.txt")
    
    return ppl_results


if __name__ == "__main__":
    main()
