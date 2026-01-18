"""
SNN Language Model - Large Scale Comparison
=============================================

Testing with larger models (500-600 neurons) and more text data.

Author: Hiroto Funasaki (roll)
Date: 2026-01-18
"""

import numpy as np
from collections import deque
import time


# ============================================
# SNN Language Model (Hybrid)
# ============================================

class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        
    def step(self, I_syn, dt=0.5):
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * dt
        self.v += dv
        
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            return 1.0
        return 0.0
    
    def get_potential(self):
        return (self.v - self.v_reset) / (self.v_thresh - self.v_reset)
    
    def reset(self):
        self.v = self.v_rest


class SNNLanguageModel:
    def __init__(self, num_neurons=500, vocab_size=128, seed=42):
        self.name = f"SNN-{num_neurons}"
        self.num_neurons = num_neurons
        self.vocab_size = vocab_size
        self.add_count = 0
        self.mult_count = 0
        
        np.random.seed(seed)
        
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        
        self.W_res = np.random.randn(num_neurons, num_neurons) * 0.5
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.4 / rho  # Higher spectral radius for larger network
        mask = np.random.rand(num_neurons, num_neurons) < 0.1
        self.W_res *= mask
        
        self.W_in = np.random.randn(num_neurons, vocab_size) * 0.3
        self.W_out = np.zeros((vocab_size, num_neurons))
        self.W_out_m = np.zeros((vocab_size, num_neurons))
        
        self.alpha = 0.01
        self.fire_rate = np.zeros(num_neurons)
        
    def step(self, char_idx):
        x = np.zeros(self.vocab_size)
        x[char_idx] = 1.0
        
        I_in = self.W_in @ x
        I_rec = self.W_res @ self.fire_rate
        I_total = I_in + I_rec + np.random.normal(0, 0.1, self.num_neurons) + 25.0
        
        spikes = np.zeros(self.num_neurons)
        potentials = np.zeros(self.num_neurons)
        
        for i, n in enumerate(self.neurons):
            spikes[i] = n.step(I_total[i])
            potentials[i] = n.get_potential()
        
        self.fire_rate = 0.8 * self.fire_rate + 0.2 * spikes
        
        self.add_count += np.sum(spikes > 0) * self.num_neurons
        self.mult_count += np.sum(spikes > 0) * self.num_neurons
        
        return spikes, potentials
    
    def predict(self, spikes, potentials):
        logits = self.W_out @ spikes + self.W_out_m @ potentials
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
    
    def train(self, target_idx, spikes, potentials):
        target = np.zeros(self.vocab_size)
        target[target_idx] = 1.0
        pred = self.predict(spikes, potentials)
        error = target - pred
        self.W_out += self.alpha * np.outer(error, spikes)
        self.W_out_m += self.alpha * np.outer(error, potentials)
        return pred
    
    def reset(self):
        for n in self.neurons:
            n.reset()
        self.fire_rate = np.zeros(self.num_neurons)
        self.add_count = 0
        self.mult_count = 0


class DNNLanguageModel:
    def __init__(self, hidden_size=500, vocab_size=128, seed=42):
        self.name = f"DNN-{hidden_size}"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.add_count = 0
        self.mult_count = 0
        
        np.random.seed(seed)
        
        self.W1 = np.random.randn(hidden_size, vocab_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(vocab_size, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.alpha = 0.005
        
    def step(self, char_idx):
        x = np.zeros(self.vocab_size)
        x[char_idx] = 1.0
        
        self.h1 = np.tanh(self.W1 @ x)
        self.h2 = np.tanh(self.W2 @ self.h + self.h1)
        self.h = self.h2
        
        self.add_count += self.hidden_size * (self.vocab_size + self.hidden_size)
        self.mult_count += self.hidden_size * (self.vocab_size + self.hidden_size)
        
        return self.h2
    
    def predict(self, hidden):
        logits = self.W3 @ hidden
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
    
    def train(self, target_idx, hidden):
        target = np.zeros(self.vocab_size)
        target[target_idx] = 1.0
        pred = self.predict(hidden)
        error = target - pred
        self.W3 += self.alpha * np.outer(error, hidden)
        return pred
    
    def reset(self):
        self.h = np.zeros(self.hidden_size)
        self.add_count = 0
        self.mult_count = 0


class LSTMLanguageModel:
    def __init__(self, hidden_size=400, vocab_size=128, seed=42):
        self.name = f"LSTM-{hidden_size}"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.add_count = 0
        self.mult_count = 0
        
        np.random.seed(seed)
        
        combined = hidden_size + vocab_size
        self.Wf = np.random.randn(hidden_size, combined) * 0.1
        self.Wi = np.random.randn(hidden_size, combined) * 0.1
        self.Wc = np.random.randn(hidden_size, combined) * 0.1
        self.Wo = np.random.randn(hidden_size, combined) * 0.1
        self.Wy = np.random.randn(vocab_size, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.alpha = 0.005
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def step(self, char_idx):
        x = np.zeros(self.vocab_size)
        x[char_idx] = 1.0
        
        hx = np.concatenate([self.h, x])
        
        f = self.sigmoid(self.Wf @ hx)
        i = self.sigmoid(self.Wi @ hx)
        c_tilde = np.tanh(self.Wc @ hx)
        o = self.sigmoid(self.Wo @ hx)
        
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        
        combined = self.hidden_size + self.vocab_size
        self.add_count += 4 * self.hidden_size * combined
        self.mult_count += 4 * self.hidden_size * combined
        
        return self.h
    
    def predict(self, hidden):
        logits = self.Wy @ hidden
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
    
    def train(self, target_idx, hidden):
        target = np.zeros(self.vocab_size)
        target[target_idx] = 1.0
        pred = self.predict(hidden)
        error = target - pred
        self.Wy += self.alpha * np.outer(error, hidden)
        return pred
    
    def reset(self):
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
        self.add_count = 0
        self.mult_count = 0


def train_model(model, text, epochs=1, add_noise=0.0):
    correct = 0
    total = 0
    loss_sum = 0
    
    for epoch in range(epochs):
        model.reset()
        
        for i in range(len(text) - 1):
            char_idx = ord(text[i]) % model.vocab_size
            next_idx = ord(text[i + 1]) % model.vocab_size
            
            if add_noise > 0:
                if np.random.random() < add_noise:
                    char_idx = np.random.randint(0, model.vocab_size)
            
            if isinstance(model, SNNLanguageModel):
                spikes, potentials = model.step(char_idx)
                pred = model.train(next_idx, spikes, potentials)
            else:
                hidden = model.step(char_idx)
                pred = model.train(next_idx, hidden)
            
            if np.argmax(pred) == next_idx:
                correct += 1
            total += 1
            loss_sum += -np.log(pred[next_idx] + 1e-10)
    
    accuracy = correct / total * 100
    perplexity = np.exp(loss_sum / total)
    energy = model.add_count + model.mult_count
    
    return accuracy, perplexity, energy


def main():
    print("=" * 70)
    print("   SNN LANGUAGE MODEL - LARGE SCALE COMPARISON")
    print("   Testing with 500-600 neurons and 15,000+ characters")
    print("=" * 70)
    
    # Much longer training text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Spiking neural networks are brain-inspired computing models.
    Unlike traditional neural networks, SNNs use discrete spikes.
    This makes them more energy efficient and biologically plausible.
    The membrane potential carries important information too.
    By combining spikes and membrane potentials, we get better results.
    Language modeling is the task of predicting the next word.
    Neural networks have revolutionized natural language processing.
    Deep learning has enabled machines to understand human language.
    The transformer architecture has become the dominant approach.
    However, transformers are computationally expensive.
    Spiking neural networks offer an alternative approach.
    They are more energy efficient and can run on neuromorphic hardware.
    The brain uses spikes to communicate between neurons.
    This is fundamentally different from traditional neural networks.
    We believe that SNNs can achieve competitive performance.
    The key is to leverage both spike counts and membrane potentials.
    This hybrid approach combines the best of both worlds.
    """ * 20  # Repeat for more data
    
    sample_text = ''.join(c for c in sample_text if 32 <= ord(c) < 128)
    
    print(f"\n  Training text: {len(sample_text)} characters")
    print(f"  Unique characters: {len(set(sample_text))}")
    
    # Model configurations
    configs = [
        ("SNN", [300, 500, 600]),
        ("DNN", [300, 500, 600]),
        ("LSTM", [200, 300, 400]),
    ]
    
    results = {}
    
    # ================================
    # Experiment 1: Scale Comparison
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: Scale Comparison (Larger Models)")
    print("=" * 70)
    
    for model_type, sizes in configs:
        for size in sizes:
            print(f"\n  Testing {model_type}-{size}...", end=" ", flush=True)
            
            start = time.time()
            
            if model_type == "SNN":
                model = SNNLanguageModel(num_neurons=size)
            elif model_type == "DNN":
                model = DNNLanguageModel(hidden_size=size)
            else:
                model = LSTMLanguageModel(hidden_size=size)
            
            acc, ppl, energy = train_model(model, sample_text, epochs=2)
            elapsed = time.time() - start
            
            results[model.name] = (acc, ppl, energy, elapsed)
            print(f"Acc={acc:.2f}%, PPL={ppl:.2f}, Time={elapsed:.1f}s")
    
    # ================================
    # Experiment 2: Noise Robustness at Large Scale
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Noise Robustness (Large Models)")
    print("=" * 70)
    
    noise_levels = [0.0, 0.10, 0.20, 0.30]
    noise_results = {}
    
    large_models = [
        ("SNN", 500),
        ("DNN", 500),
        ("LSTM", 300),
    ]
    
    for model_type, size in large_models:
        name = f"{model_type}-{size}"
        noise_results[name] = []
        
        for noise in noise_levels:
            if model_type == "SNN":
                model = SNNLanguageModel(num_neurons=size)
            elif model_type == "DNN":
                model = DNNLanguageModel(hidden_size=size)
            else:
                model = LSTMLanguageModel(hidden_size=size)
            
            acc, ppl, energy = train_model(model, sample_text, epochs=2, add_noise=noise)
            noise_results[name].append((noise, acc, ppl))
    
    print(f"\n  {'Model':<12}", end="")
    for noise in noise_levels:
        print(f" {noise*100:>6.0f}%", end="")
    print("  | Degradation")
    print("-" * 70)
    
    for name in noise_results.keys():
        print(f"  {name:<12}", end="")
        for noise, acc, ppl in noise_results[name]:
            print(f" {acc:>6.2f}%", end="")
        
        base = noise_results[name][0][1]
        noisy = noise_results[name][-1][1]
        degradation = base - noisy
        print(f"  | {degradation:>+.2f}%")
    
    # ================================
    # Experiment 3: Energy Efficiency at Large Scale
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: Energy Efficiency (Large Models)")
    print("=" * 70)
    
    print(f"\n  {'Model':<12} | {'Accuracy':<10} | {'Operations':<15} | {'Efficiency':<12}")
    print("-" * 60)
    
    efficiency_scores = []
    for name, (acc, ppl, energy, elapsed) in results.items():
        eff = acc / (energy / 1e6) if energy > 0 else 0
        efficiency_scores.append((name, acc, energy, eff))
        print(f"  {name:<12} | {acc:>8.2f}% | {energy/1e6:>13.2f}M | {eff:>10.4f}")
    
    # ================================
    # Summary
    # ================================
    print("\n" + "=" * 70)
    print("   SUMMARY: SNN STRENGTHS AT LARGE SCALE")
    print("=" * 70)
    
    # Find best SNN
    snn_results = {k: v for k, v in results.items() if k.startswith("SNN")}
    dnn_results = {k: v for k, v in results.items() if k.startswith("DNN")}
    lstm_results = {k: v for k, v in results.items() if k.startswith("LSTM")}
    
    best_snn = max(snn_results.items(), key=lambda x: x[1][0])
    best_dnn = max(dnn_results.items(), key=lambda x: x[1][0])
    best_lstm = max(lstm_results.items(), key=lambda x: x[1][0])
    
    print(f"\n  Best SNN: {best_snn[0]} - Acc={best_snn[1][0]:.2f}%")
    print(f"  Best DNN: {best_dnn[0]} - Acc={best_dnn[1][0]:.2f}%")
    print(f"  Best LSTM: {best_lstm[0]} - Acc={best_lstm[1][0]:.2f}%")
    
    # Noise robustness
    print("\n  Noise Robustness (0% -> 30%):")
    for name in ["SNN-500", "DNN-500", "LSTM-300"]:
        if name in noise_results:
            base = noise_results[name][0][1]
            noisy = noise_results[name][-1][1]
            degradation = base - noisy
            symbol = "✅ ROBUST!" if degradation < 3.0 else "⚠️ Degraded"
            print(f"    {name}: {degradation:>+.2f}% {symbol}")
    
    # Energy efficiency
    print("\n  Energy Efficiency (Acc per Million Ops):")
    snn_effs = [(n, e) for n, a, en, e in efficiency_scores if n.startswith("SNN")]
    dnn_effs = [(n, e) for n, a, en, e in efficiency_scores if n.startswith("DNN")]
    
    if snn_effs and dnn_effs:
        best_snn_eff = max(snn_effs, key=lambda x: x[1])
        best_dnn_eff = max(dnn_effs, key=lambda x: x[1])
        ratio = best_snn_eff[1] / best_dnn_eff[1] if best_dnn_eff[1] > 0 else 0
        print(f"    SNN: {best_snn_eff[0]} = {best_snn_eff[1]:.4f}")
        print(f"    DNN: {best_dnn_eff[0]} = {best_dnn_eff[1]:.4f}")
        print(f"    → SNN is {ratio:.1f}x more energy efficient!")
    
    findings = []
    
    # Check if SNN is more robust
    snn_deg = noise_results.get("SNN-500", [(0, 0, 0)])[0][1] - noise_results.get("SNN-500", [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])[-1][1]
    dnn_deg = noise_results.get("DNN-500", [(0, 0, 0)])[0][1] - noise_results.get("DNN-500", [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])[-1][1]
    
    if snn_deg < dnn_deg:
        findings.append("✅ SNN is MORE ROBUST to noise at large scale!")
    
    if ratio > 1.5:
        findings.append(f"✅ SNN is {ratio:.1f}x MORE ENERGY EFFICIENT!")
    
    print("\n  KEY FINDINGS:")
    for f in findings:
        print(f"    {f}")
    
    # Save results
    with open("results/large_scale_comparison.txt", "w", encoding="utf-8") as f:
        f.write("SNN Language Model - Large Scale Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Training text: {len(sample_text)} characters\n\n")
        
        f.write("Accuracy Results:\n")
        for name, (acc, ppl, energy, elapsed) in results.items():
            f.write(f"  {name}: Acc={acc:.2f}%, PPL={ppl:.2f}, Time={elapsed:.1f}s\n")
        
        f.write("\nNoise Robustness:\n")
        for name in noise_results.keys():
            base = noise_results[name][0][1]
            noisy = noise_results[name][-1][1]
            f.write(f"  {name}: {base - noisy:+.2f}% degradation\n")
        
        f.write("\nEnergy Efficiency:\n")
        for name, acc, energy, eff in efficiency_scores:
            f.write(f"  {name}: {eff:.4f} acc/MOps\n")
        
        f.write("\nFindings:\n")
        for finding in findings:
            f.write(f"  {finding}\n")
    
    print("\n  Results saved to: results/large_scale_comparison.txt")


if __name__ == "__main__":
    main()
