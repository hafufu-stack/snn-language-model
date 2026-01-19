"""
SNN Language Model - Full Comparison Study
============================================

Comparing SNN vs DNN vs LSTM for character-level language modeling
to find where SNN excels!

Experiments:
1. Architecture comparison (SNN vs DNN vs LSTM)
2. Noise robustness (which model degrades gracefully?)
3. Sample efficiency (who learns faster with less data?)
4. Energy efficiency (proxy: computation count)

Author: Hiroto Funasaki (roll)
Date: 2026-01-18
"""

import numpy as np
from collections import deque
import time
from multiprocessing import Pool, cpu_count


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
    def __init__(self, num_neurons=150, vocab_size=128):
        self.name = f"SNN-{num_neurons}"
        self.num_neurons = num_neurons
        self.vocab_size = vocab_size
        self.add_count = 0  # For energy estimation
        self.mult_count = 0
        
        np.random.seed(42)
        
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        
        self.W_res = np.random.randn(num_neurons, num_neurons) * 0.5
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.2 / rho
        mask = np.random.rand(num_neurons, num_neurons) < 0.1
        self.W_res *= mask
        
        self.W_in = np.random.randn(num_neurons, vocab_size) * 0.5
        self.W_out = np.zeros((vocab_size, num_neurons))
        self.W_out_m = np.zeros((vocab_size, num_neurons))
        
        self.alpha = 0.02
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
        
        # Energy count (sparse!)
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


# ============================================
# DNN Language Model
# ============================================

class DNNLanguageModel:
    def __init__(self, hidden_size=150, vocab_size=128):
        self.name = f"DNN-{hidden_size}"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.add_count = 0
        self.mult_count = 0
        
        np.random.seed(42)
        
        self.W1 = np.random.randn(hidden_size, vocab_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(vocab_size, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.alpha = 0.01
        
    def step(self, char_idx):
        x = np.zeros(self.vocab_size)
        x[char_idx] = 1.0
        
        self.h1 = np.tanh(self.W1 @ x)
        self.h2 = np.tanh(self.W2 @ self.h + self.h1)
        self.h = self.h2
        
        # Energy count (dense!)
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


# ============================================
# LSTM Language Model
# ============================================

class LSTMLanguageModel:
    def __init__(self, hidden_size=100, vocab_size=128):
        self.name = f"LSTM-{hidden_size}"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.add_count = 0
        self.mult_count = 0
        
        np.random.seed(42)
        
        combined = hidden_size + vocab_size
        self.Wf = np.random.randn(hidden_size, combined) * 0.1
        self.Wi = np.random.randn(hidden_size, combined) * 0.1
        self.Wc = np.random.randn(hidden_size, combined) * 0.1
        self.Wo = np.random.randn(hidden_size, combined) * 0.1
        self.Wy = np.random.randn(vocab_size, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.alpha = 0.01
        
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
        
        # Energy count (very dense!)
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


# ============================================
# Experiments
# ============================================

def train_model(model, text, epochs=1, add_noise=0.0):
    """Train and evaluate a model"""
    correct = 0
    total = 0
    loss_sum = 0
    
    for epoch in range(epochs):
        model.reset()
        
        for i in range(len(text) - 1):
            char_idx = ord(text[i]) % model.vocab_size
            next_idx = ord(text[i + 1]) % model.vocab_size
            
            # Add noise to input
            if add_noise > 0:
                if np.random.random() < add_noise:
                    char_idx = np.random.randint(0, model.vocab_size)
            
            # Forward
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


def run_experiment(args):
    """Run single experiment (for parallel processing)"""
    model_type, size, text, epochs, noise, seed = args
    
    np.random.seed(seed)
    
    if model_type == "SNN":
        model = SNNLanguageModel(num_neurons=size)
    elif model_type == "DNN":
        model = DNNLanguageModel(hidden_size=size)
    elif model_type == "LSTM":
        model = LSTMLanguageModel(hidden_size=size)
    
    acc, ppl, energy = train_model(model, text, epochs=epochs, add_noise=noise)
    
    return (model.name, acc, ppl, energy)


def main():
    print("=" * 70)
    print("   SNN LANGUAGE MODEL - ARCHITECTURE COMPARISON")
    print("   Finding where SNN excels!")
    print("=" * 70)
    
    # Training text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Spiking neural networks are brain-inspired computing models.
    Unlike traditional neural networks, SNNs use discrete spikes.
    This makes them more energy efficient and biologically plausible.
    The membrane potential carries important information too.
    By combining spikes and membrane potentials, we get better results.
    Language modeling is the task of predicting the next word.
    Neural networks have revolutionized natural language processing.
    """ * 10
    
    sample_text = ''.join(c for c in sample_text if 32 <= ord(c) < 128)
    
    print(f"\n  Training text: {len(sample_text)} chars")
    
    # ================================
    # Experiment 1: Architecture Comparison
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: Architecture Comparison")
    print("=" * 70)
    
    models_config = [
        ("SNN", 150),
        ("DNN", 150),
        ("LSTM", 100),
    ]
    
    results = {}
    
    for model_type, size in models_config:
        tasks = [(model_type, size, sample_text, 3, 0.0, seed) for seed in range(3)]
        
        all_results = []
        for task in tasks:
            result = run_experiment(task)
            all_results.append(result)
        
        name = all_results[0][0]
        avg_acc = np.mean([r[1] for r in all_results])
        avg_ppl = np.mean([r[2] for r in all_results])
        avg_energy = np.mean([r[3] for r in all_results])
        
        results[name] = (avg_acc, avg_ppl, avg_energy)
        print(f"  {name}: Acc={avg_acc:.2f}%, PPL={avg_ppl:.2f}, Energy={avg_energy/1e6:.2f}M ops")
    
    # ================================
    # Experiment 2: Noise Robustness
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Noise Robustness")
    print("=" * 70)
    
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    noise_results = {name: [] for name in ["SNN-150", "DNN-150", "LSTM-100"]}
    
    for noise in noise_levels:
        for model_type, size in models_config:
            result = run_experiment((model_type, size, sample_text, 3, noise, 42))
            noise_results[result[0]].append((noise, result[1], result[2]))
    
    print(f"\n  {'Noise':<8}", end="")
    for name in noise_results.keys():
        print(f" {name:<15}", end="")
    print()
    print("-" * 60)
    
    for i, noise in enumerate(noise_levels):
        print(f"  {noise*100:>5.0f}%  ", end="")
        for name in noise_results.keys():
            acc = noise_results[name][i][1]
            print(f" {acc:>12.2f}%  ", end="")
        print()
    
    # Calculate degradation
    print("\n  Degradation from 0% to 30% noise:")
    for name in noise_results.keys():
        base = noise_results[name][0][1]
        noisy = noise_results[name][-1][1]
        degradation = base - noisy
        print(f"    {name}: {degradation:+.2f}% (Lower = More Robust)")
    
    # ================================
    # Experiment 3: Sample Efficiency
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: Sample Efficiency (Learning Speed)")
    print("=" * 70)
    
    text_sizes = [200, 500, 1000, 2000, 4000]
    sample_results = {name: [] for name in ["SNN-150", "DNN-150", "LSTM-100"]}
    
    for size in text_sizes:
        for model_type, param_size in models_config:
            result = run_experiment((model_type, param_size, sample_text[:size], 3, 0.0, 42))
            sample_results[result[0]].append((size, result[1], result[2]))
    
    print(f"\n  {'Text Size':<10}", end="")
    for name in sample_results.keys():
        print(f" {name:<15}", end="")
    print()
    print("-" * 60)
    
    for i, size in enumerate(text_sizes):
        print(f"  {size:>8}  ", end="")
        for name in sample_results.keys():
            acc = sample_results[name][i][1]
            print(f" {acc:>12.2f}%  ", end="")
        print()
    
    # ================================
    # Experiment 4: Energy Efficiency
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 4: Energy Efficiency")
    print("=" * 70)
    
    print("\n  Model       | Accuracy | Operations | Efficiency (Acc/MOps)")
    print("-" * 60)
    
    efficiency_ranking = []
    for name, (acc, ppl, energy) in results.items():
        efficiency = acc / (energy / 1e6)
        efficiency_ranking.append((name, acc, energy, efficiency))
        print(f"  {name:<12} | {acc:>7.2f}% | {energy/1e6:>9.2f}M | {efficiency:>10.4f}")
    
    # ================================
    # Summary
    # ================================
    print("\n" + "=" * 70)
    print("   SUMMARY: WHERE DOES SNN EXCEL?")
    print("=" * 70)
    
    snn_acc = results["SNN-150"][0]
    dnn_acc = results["DNN-150"][0]
    lstm_acc = results["LSTM-100"][0]
    
    snn_degradation = noise_results["SNN-150"][0][1] - noise_results["SNN-150"][-1][1]
    dnn_degradation = noise_results["DNN-150"][0][1] - noise_results["DNN-150"][-1][1]
    lstm_degradation = noise_results["LSTM-100"][0][1] - noise_results["LSTM-100"][-1][1]
    
    findings = []
    
    # Check noise robustness
    if snn_degradation < dnn_degradation and snn_degradation < lstm_degradation:
        findings.append("✅ SNN is MOST ROBUST to input noise!")
    
    # Check energy efficiency
    snn_eff = efficiency_ranking[0][3] if efficiency_ranking[0][0] == "SNN-150" else efficiency_ranking[1][3] if efficiency_ranking[1][0] == "SNN-150" else efficiency_ranking[2][3]
    dnn_eff = efficiency_ranking[0][3] if efficiency_ranking[0][0] == "DNN-150" else efficiency_ranking[1][3] if efficiency_ranking[1][0] == "DNN-150" else efficiency_ranking[2][3]
    lstm_eff = efficiency_ranking[0][3] if efficiency_ranking[0][0] == "LSTM-100" else efficiency_ranking[1][3] if efficiency_ranking[1][0] == "LSTM-100" else efficiency_ranking[2][3]
    
    if snn_eff > dnn_eff and snn_eff > lstm_eff:
        findings.append("✅ SNN is MOST ENERGY EFFICIENT!")
    
    # Check accuracy
    if snn_acc >= dnn_acc and snn_acc >= lstm_acc:
        findings.append("✅ SNN achieves BEST or EQUAL ACCURACY!")
    
    for finding in findings:
        print(f"\n  {finding}")
    
    if not findings:
        print("\n  🤔 No clear SNN advantage found. Try more experiments!")
    
    # Save results
    with open("results/architecture_comparison.txt", "w", encoding="utf-8") as f:
        f.write("SNN Language Model - Architecture Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Accuracy:\n")
        for name, (acc, ppl, energy) in results.items():
            f.write(f"  {name}: {acc:.2f}%\n")
        
        f.write("\nNoise Robustness (degradation at 30% noise):\n")
        f.write(f"  SNN: {snn_degradation:.2f}%\n")
        f.write(f"  DNN: {dnn_degradation:.2f}%\n")
        f.write(f"  LSTM: {lstm_degradation:.2f}%\n")
        
        f.write("\nEnergy Efficiency:\n")
        for name, acc, energy, eff in efficiency_ranking:
            f.write(f"  {name}: {eff:.4f} acc/MOps\n")
        
        f.write("\nFindings:\n")
        for finding in findings:
            f.write(f"  {finding}\n")
    
    print("\n  Results saved to: results/architecture_comparison.txt")


if __name__ == "__main__":
    main()
