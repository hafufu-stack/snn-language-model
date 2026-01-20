"""
Super Ultimate++ : Further Improvements
=========================================

Trying more improvements:
1. Deeper Progressive (6 phases instead of 4)
2. Larger models (600 neurons)
3. Multi-head attention
4. Adaptive learning rate
5. Residual connections

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


class SuperUltimatePlusSNN:
    """
    Super Ultimate++ with:
    - 6-phase progressive training
    - 600 neurons
    - Multi-head attention (2 heads)
    - Residual connections
    - Adaptive learning rate
    """
    
    def __init__(self, vocab_size, hidden_size=600, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.active_size = hidden_size // 6  # Deeper progressive
        self.n_heads = 2
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.35
        self.time_decay = np.random.uniform(0.75, 0.95, hidden_size)
        
        # RWKV
        self.W_key = np.random.randn(hidden_size, hidden_size) * 0.07
        self.W_value = np.random.randn(hidden_size, hidden_size) * 0.07
        self.W_gate = np.random.randn(hidden_size, hidden_size) * 0.07
        
        # Multi-head attention
        head_dim = hidden_size // 4 // self.n_heads
        self.W_Q = [np.random.randn(hidden_size, head_dim) * 0.1 for _ in range(self.n_heads)]
        self.W_K = [np.random.randn(hidden_size, head_dim) * 0.1 for _ in range(self.n_heads)]
        self.W_V = [np.random.randn(hidden_size, hidden_size // self.n_heads) * 0.1 for _ in range(self.n_heads)]
        
        # Reservoir
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        self.mask = np.random.rand(hidden_size, hidden_size) < 0.08
        
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        self.lr = 0.12
        self.epoch = 0
    
    def grow(self):
        self.active_size = min(self.active_size + self.hidden_size // 6, self.hidden_size)
    
    def get_lr(self):
        # Adaptive LR: decay over epochs
        return self.lr * (0.95 ** self.epoch)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def multi_head_attention(self, query, history, h):
        """Multi-head attention"""
        if len(history) <= 1:
            return np.zeros(h)
        
        H = np.array(history[-7:])  # Longer context
        contexts = []
        
        for head in range(self.n_heads):
            Q = query @ self.W_Q[head][:h, :]
            K = H @ self.W_K[head][:h, :]
            scores = K @ Q / np.sqrt(Q.shape[0] + 1)
            attn = self.softmax(scores)
            V = H @ self.W_V[head][:h, :h // self.n_heads]
            contexts.append(attn @ V)
        
        return np.concatenate(contexts)[:h]
    
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
            
            # Residual connection
            channel_out = channel_out + I_in * 0.1
            
            state = mixed
            history.append(channel_out.copy())
            
            # Multi-head attention
            context = self.multi_head_attention(channel_out, history, h)
            if context.shape[0] > 0:
                channel_out = channel_out + context * 0.15
            
            # Spiking dynamics
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
        self.W_out += self.get_lr() * np.outer(features, target_vec - probs)
        return -np.log(probs[target] + 1e-10)


class SuperUltimateSNN:
    """Original Super Ultimate (baseline)"""
    
    def __init__(self, vocab_size, hidden_size=500, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.active_size = hidden_size // 4
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.4
        self.time_decay = np.random.uniform(0.7, 0.95, hidden_size)
        self.W_key = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_value = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_gate = np.random.randn(hidden_size, hidden_size) * 0.08
        self.W_Q = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_K = np.random.randn(hidden_size, hidden_size // 4) * 0.1
        self.W_V = np.random.randn(hidden_size, hidden_size) * 0.1
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
            
            mixed = I_in * (1 - self.time_decay[:h]) + state * self.time_decay[:h]
            key = mixed @ self.W_key[:h, :h]
            value = mixed @ self.W_value[:h, :h]
            gate = self.sigmoid(mixed @ self.W_gate[:h, :h])
            channel_out = gate * (key * value / (np.abs(key).max() + 1e-10))
            
            state = mixed
            history.append(channel_out.copy())
            
            if len(history) > 1:
                H = np.array(history[-5:])
                Q = channel_out @ self.W_Q[:h, :]
                K = H @ self.W_K[:h, :]
                scores = K @ Q / np.sqrt(h // 4 + 1)
                attn = self.softmax(scores)
                V = H @ self.W_V[:h, :h]
                context = attn @ V
                channel_out = channel_out + context * 0.2
            
            for t in range(time_steps):
                spiking = (v > 1.0).astype(float)
                I_rec = W_res_tern @ spiking * 0.3
                v = v * 0.9 + channel_out * 0.5 + I_rec
                spike_counts += spiking
                v[spiking > 0] = 0
        
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
    base_texts = [
        "once upon a time there was a small village nestled in the mountains ",
        "the villagers lived simple lives farming and trading with neighbors ",
        "neural networks learn patterns from vast amounts of training data ",
        "spiking networks mimic the brain using discrete pulses of activity ",
        "transformers use attention mechanisms to capture long range patterns ",
        "deep learning has revolutionized artificial intelligence research ",
        "machine learning algorithms can recognize images speech and text ",
        "natural language processing enables computers to understand humans ",
        "reinforcement learning teaches agents through trial and error ",
        "convolutional networks excel at image classification tasks ",
    ]
    text = ""
    while len(text) < 50000:
        for base in base_texts:
            text += base
    return text.lower()


def train_worker(args):
    model_class, vocab_size, train_seq, train_tgt, seed, epochs = args
    
    if model_class == 'super_plus':
        model = SuperUltimatePlusSNN(vocab_size, 600, seed)
        phases = 6
    else:
        model = SuperUltimateSNN(vocab_size, 500, seed)
        phases = 4
    
    for phase in range(phases):
        model.epoch = phase
        for _ in range(epochs // phases):
            for i in range(0, len(train_seq), 3):
                model.train_step(train_seq[i], train_tgt[i])
        model.grow()
    
    return model


def main():
    n_parallel = min(24, cpu_count())
    
    print("=" * 70)
    print("   SUPER ULTIMATE++ EXPERIMENT")
    print(f"   Using {n_parallel} parallel workers")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text):,} chars")
    print(f"  Vocab: {vocab_size} chars")
    print(f"  Train: {n_train:,}, Test: {n - n_train:,}")
    
    epochs = 18
    models_per_type = n_parallel // 2
    
    all_args = []
    for i in range(models_per_type):
        all_args.append(('super_plus', vocab_size, train_seq, train_tgt, 42 + i, epochs))
        all_args.append(('super_ultimate', vocab_size, train_seq, train_tgt, 100 + i, epochs))
    
    print(f"\n  Training {len(all_args)} models for {epochs} epochs...")
    
    t0 = time.time()
    with Pool(n_parallel) as pool:
        all_models = pool.map(train_worker, all_args)
    train_time = time.time() - t0
    
    plus_models = [m for i, m in enumerate(all_models) if i % 2 == 0]
    base_models = [m for i, m in enumerate(all_models) if i % 2 == 1]
    
    print("\n  Testing...")
    
    def test_models(models):
        all_losses = []
        for model in models:
            losses = []
            for i in range(len(test_seq)):
                probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
            all_losses.append(np.mean(losses))
        return all_losses
    
    plus_losses = test_models(plus_models)
    base_losses = test_models(base_models)
    
    plus_ppl = np.exp(np.mean(plus_losses))
    base_ppl = np.exp(np.mean(base_losses))
    
    plus_std = np.std([np.exp(l) for l in plus_losses])
    base_std = np.std([np.exp(l) for l in base_losses])
    
    gap = (plus_ppl - base_ppl) / base_ppl * 100
    
    print("\n" + "=" * 70)
    print("   SUPER ULTIMATE++ RESULTS")
    print("=" * 70)
    
    print(f"""
    ┌────────────────────────────┬────────────┬────────────┬──────────┐
    │ Model                      │ PPL (±std) │ vs Base    │          │
    ├────────────────────────────┼────────────┼────────────┼──────────┤
    │ Super Ultimate++ (600n)    │ {plus_ppl:6.2f}±{plus_std:4.2f} │ {gap:+10.1f}% │ {'✅ WINS!' if plus_ppl < base_ppl else '❌ LOSES'} │
    │ Super Ultimate (500n)      │ {base_ppl:6.2f}±{base_std:4.2f} │   baseline │          │
    └────────────────────────────┴────────────┴────────────┴──────────┘
    
    Training time: {train_time:.1f}s ({train_time/60:.1f} min)
    """)
    
    if plus_ppl < base_ppl:
        print(f"  🚀 Super Ultimate++ beats Super Ultimate by {-gap:.1f}%!")
    else:
        print(f"  ℹ️ Super Ultimate++ did not beat Super Ultimate (may need tuning)")
    
    with open("results/super_ultimate_plus_results.txt", "w", encoding="utf-8") as f:
        f.write("Super Ultimate++ Experiment\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Super Ultimate++ PPL: {plus_ppl:.2f} +/- {plus_std:.2f}\n")
        f.write(f"Super Ultimate PPL: {base_ppl:.2f} +/- {base_std:.2f}\n")
        f.write(f"Gap: {gap:+.1f}%\n\n")
        f.write(f"Winner: {'Super Ultimate++' if plus_ppl < base_ppl else 'Super Ultimate'}\n")
    
    print("\n  Results saved to: results/super_ultimate_plus_results.txt")


if __name__ == "__main__":
    main()
