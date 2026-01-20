"""
Large-Scale Validation: Super Ultimate SNN
============================================

Testing on 100,000+ characters to validate the discovery!

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


class SuperUltimateSNN:
    """Everything combined: BitNet + RWKV + Hybrid + Progressive + Attention"""
    
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


def get_large_text():
    """Generate 100,000+ character dataset"""
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
        "recurrent networks process sequential data like time series ",
        "generative models can create new images text and music ",
        "edge computing brings intelligence closer to the data source ",
        "neuromorphic chips implement brain inspired computing hardware ",
        "energy efficiency is critical for sustainable computing ",
        "quantum computing promises exponential speedups for some problems ",
        "federated learning enables privacy preserving model training ",
        "transfer learning reuses knowledge from pretrained models ",
        "attention mechanisms weigh the importance of different inputs ",
        "batch normalization stabilizes training of deep networks ",
    ]
    
    text = ""
    while len(text) < 120000:
        for base in base_texts:
            text += base
            if len(text) >= 120000:
                break
    
    return text.lower()


def train_worker(args):
    model_class, vocab_size, train_seq, train_tgt, seed, epochs = args
    
    if model_class == 'super_ultimate':
        model = SuperUltimateSNN(vocab_size, 500, seed)
        for phase in range(4):
            for _ in range(epochs // 4):
                for i in range(0, len(train_seq), 3):
                    model.train_step(train_seq[i], train_tgt[i])
            model.grow()
    else:
        model = StandardSNN(vocab_size, 200, seed)
        for _ in range(epochs):
            for i in range(0, len(train_seq), 3):
                model.train_step(train_seq[i], train_tgt[i])
    
    return model


def main():
    n_parallel = min(24, cpu_count())  # Use ALL cores!
    
    print("=" * 70)
    print("   LARGE-SCALE VALIDATION: SUPER ULTIMATE SNN")
    print(f"   Target: 100,000+ characters")
    print(f"   Using {n_parallel} parallel workers")
    print("=" * 70)
    
    text = get_large_text()
    sequences, targets, vocab_size = prepare_data(text, seq_length=30)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    print(f"\n  Data: {len(text):,} chars (100K+)")
    print(f"  Vocab: {vocab_size} chars")
    print(f"  Train samples: {n_train:,}")
    print(f"  Test samples: {n - n_train:,}")
    
    epochs = 15
    models_per_type = n_parallel // 2
    
    all_args = []
    for i in range(models_per_type):
        all_args.append(('super_ultimate', vocab_size, train_seq, train_tgt, 42 + i, epochs))
        all_args.append(('standard', vocab_size, train_seq, train_tgt, 100 + i, epochs))
    
    print(f"\n  Training {len(all_args)} models for {epochs} epochs...")
    print("  (This may take a few minutes...)")
    
    t0 = time.time()
    with Pool(n_parallel) as pool:
        all_models = pool.map(train_worker, all_args)
    train_time = time.time() - t0
    
    super_models = [m for i, m in enumerate(all_models) if i % 2 == 0]
    standard_models = [m for i, m in enumerate(all_models) if i % 2 == 1]
    
    print("\n  Testing...")
    
    def test_models(models, name):
        all_losses = []
        for model in models:
            losses = []
            for i in range(len(test_seq)):
                probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
            all_losses.append(np.mean(losses))
        return all_losses
    
    super_losses = test_models(super_models, "Super Ultimate")
    std_losses = test_models(standard_models, "Standard")
    
    super_ppl = np.exp(np.mean(super_losses))
    std_ppl = np.exp(np.mean(std_losses))
    
    super_ppl_std = np.std([np.exp(l) for l in super_losses])
    std_ppl_std = np.std([np.exp(l) for l in std_losses])
    
    gap = (super_ppl - std_ppl) / std_ppl * 100
    
    print("\n" + "=" * 70)
    print("   LARGE-SCALE VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"""
    Dataset: {len(text):,} characters (100K+)
    
    ┌────────────────────────────┬────────────┬────────────┬──────────┐
    │ Model                      │ PPL (±std) │ vs Standard│          │
    ├────────────────────────────┼────────────┼────────────┼──────────┤
    │ Super Ultimate (500n)      │ {super_ppl:6.2f}±{super_ppl_std:4.2f} │ {gap:+10.1f}% │ {'✅ WINS!' if super_ppl < std_ppl else '❌ LOSES'} │
    │ Standard SNN (200n)        │ {std_ppl:6.2f}±{std_ppl_std:4.2f} │   baseline │          │
    └────────────────────────────┴────────────┴────────────┴──────────┘
    
    Training time: {train_time:.1f}s ({train_time/60:.1f} min)
    """)
    
    if super_ppl < std_ppl:
        print(f"  🎉 VALIDATED! Super Ultimate beats Standard by {-gap:.1f}%")
        print("  📝 This is a REAL discovery!")
    else:
        print(f"  ⚠️ Super Ultimate did NOT beat Standard on large data")
        print("  📝 The small-data results may have been overfitting")
    
    with open("results/large_scale_validation.txt", "w", encoding="utf-8") as f:
        f.write("Large-Scale Validation: Super Ultimate SNN\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {len(text):,} characters\n")
        f.write(f"Train: {n_train:,}, Test: {n - n_train:,}\n\n")
        f.write(f"Super Ultimate PPL: {super_ppl:.2f} +/- {super_ppl_std:.2f}\n")
        f.write(f"Standard SNN PPL: {std_ppl:.2f} +/- {std_ppl_std:.2f}\n")
        f.write(f"Gap: {gap:+.1f}%\n\n")
        f.write(f"Validated: {'YES' if super_ppl < std_ppl else 'NO'}\n")
    
    print("\n  Results saved to: results/large_scale_validation.txt")
    
    return super_ppl, std_ppl


if __name__ == "__main__":
    main()
