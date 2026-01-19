"""
SNN Language Model - Hybrid Spike + Membrane Potential Approach
================================================================

A character-level language model using Spiking Neural Networks.
Key innovation: Uses both spike counts AND membrane potentials
for prediction, inspired by brain-inspired decision making.

Author: Hiroto Funasaki (roll)
Date: 2026-01-18
"""

import numpy as np
from collections import deque
import time


class LIFNeuron:
    """Leaky Integrate-and-Fire Neuron with membrane potential access"""
    
    def __init__(self, tau=20.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        self.spike_count = 0
        
    def step(self, I_syn, dt=0.5):
        """Update neuron state and return spike (0 or 1)"""
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * dt
        self.v += dv
        
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.spike_count += 1
            return 1.0
        return 0.0
    
    def get_membrane_potential(self):
        """Return normalized membrane potential (0-1 range)"""
        # Normalize to 0-1 range
        return (self.v - self.v_reset) / (self.v_thresh - self.v_reset)
    
    def reset(self):
        self.v = self.v_rest
        self.spike_count = 0


class SNNLanguageModel:
    """
    SNN-based Language Model using Reservoir Computing
    
    Key features:
    1. Character-level prediction (vocab size = 256)
    2. Hybrid evaluation: spike_count + membrane_potential
    3. Online learning with LMS
    """
    
    def __init__(self, num_neurons=200, vocab_size=128, context_size=32):
        self.num_neurons = num_neurons
        self.vocab_size = vocab_size
        self.context_size = context_size
        
        np.random.seed(42)
        
        # Reservoir
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        
        # Reservoir weights (Edge of Chaos)
        self.W_res = np.random.randn(num_neurons, num_neurons) * 0.5
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.2 / rho  # Spectral radius = 1.2
        
        # Sparse connectivity
        mask = np.random.rand(num_neurons, num_neurons) < 0.1
        self.W_res *= mask
        
        # Input weights (character -> neurons)
        self.W_in = np.random.randn(num_neurons, vocab_size) * 0.5
        
        # Output weights (TRADITIONAL: spike count only)
        self.W_out_spike = np.zeros((vocab_size, num_neurons))
        
        # Output weights (HYBRID: spike + membrane potential)
        self.W_out_membrane = np.zeros((vocab_size, num_neurons))
        
        # Learning rate
        self.alpha = 0.01
        
        # State
        self.fire_rate = np.zeros(num_neurons)
        self.context = deque(maxlen=context_size)
        
    def char_to_onehot(self, char):
        """Convert character to one-hot vector"""
        idx = ord(char) % self.vocab_size
        vec = np.zeros(self.vocab_size)
        vec[idx] = 1.0
        return vec
    
    def onehot_to_char(self, vec):
        """Convert one-hot vector to character"""
        idx = np.argmax(vec)
        return chr(idx)
    
    def step(self, char):
        """Process one character and return new state"""
        # Convert to input
        x = self.char_to_onehot(char)
        
        # Input current
        I_in = self.W_in @ x
        
        # Recurrent current
        I_rec = self.W_res @ self.fire_rate
        
        # Total current with noise
        I_total = I_in + I_rec + np.random.normal(0, 0.1, self.num_neurons)
        
        # Update neurons
        spikes = np.zeros(self.num_neurons)
        membrane_potentials = np.zeros(self.num_neurons)
        
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.step(I_total[i] + 25.0)  # Bias
            membrane_potentials[i] = neuron.get_membrane_potential()
        
        # Update fire rate (exponential moving average)
        self.fire_rate = 0.8 * self.fire_rate + 0.2 * spikes
        
        # Store context
        self.context.append(char)
        
        return spikes, membrane_potentials
    
    def predict_traditional(self, spikes):
        """Traditional prediction: spike count only"""
        logits = self.W_out_spike @ spikes
        return self.softmax(logits)
    
    def predict_hybrid(self, spikes, membrane_potentials):
        """Hybrid prediction: spike + membrane potential"""
        # Combine spike and membrane information
        logits = (self.W_out_spike @ spikes + 
                  self.W_out_membrane @ membrane_potentials)
        return self.softmax(logits)
    
    def softmax(self, x):
        """Stable softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def train_step(self, target_char, spikes, membrane_potentials, use_hybrid=True):
        """Update weights using LMS"""
        target_vec = self.char_to_onehot(target_char)
        
        if use_hybrid:
            pred = self.predict_hybrid(spikes, membrane_potentials)
            error = target_vec - pred
            
            # Update both weight matrices
            self.W_out_spike += self.alpha * np.outer(error, spikes)
            self.W_out_membrane += self.alpha * np.outer(error, membrane_potentials)
        else:
            pred = self.predict_traditional(spikes)
            error = target_vec - pred
            
            # Update spike weights only
            self.W_out_spike += self.alpha * np.outer(error, spikes)
        
        return np.argmax(pred), np.argmax(target_vec)
    
    def reset(self):
        """Reset all neurons"""
        for neuron in self.neurons:
            neuron.reset()
        self.fire_rate = np.zeros(self.num_neurons)
        self.context.clear()


def train_and_evaluate(model, text, use_hybrid=True, epochs=1):
    """Train model on text and evaluate perplexity"""
    
    correct = 0
    total = 0
    loss_sum = 0
    
    for epoch in range(epochs):
        model.reset()
        
        for i in range(len(text) - 1):
            current_char = text[i]
            next_char = text[i + 1]
            
            # Forward pass
            spikes, membrane_potentials = model.step(current_char)
            
            # Train
            pred_idx, target_idx = model.train_step(
                next_char, spikes, membrane_potentials, use_hybrid=use_hybrid
            )
            
            if pred_idx == target_idx:
                correct += 1
            total += 1
            
            # Cross-entropy loss
            if use_hybrid:
                probs = model.predict_hybrid(spikes, membrane_potentials)
            else:
                probs = model.predict_traditional(spikes)
            
            loss_sum += -np.log(probs[target_idx] + 1e-10)
    
    accuracy = correct / total * 100
    perplexity = np.exp(loss_sum / total)
    
    return accuracy, perplexity


def main():
    print("=" * 70)
    print("   SNN LANGUAGE MODEL - HYBRID EXPERIMENT")
    print("   Comparing Spike-only vs Spike+Membrane approaches")
    print("=" * 70)
    
    # Sample text for training
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Spiking neural networks are brain-inspired computing models.
    Unlike traditional neural networks, SNNs use discrete spikes.
    This makes them more energy efficient and biologically plausible.
    The membrane potential carries important information too.
    By combining spikes and membrane potentials, we get better results.
    """ * 5  # Repeat for more training data
    
    # Clean text (keep only ASCII printable)
    sample_text = ''.join(c for c in sample_text if 32 <= ord(c) < 128)
    
    print(f"\n  Training text length: {len(sample_text)} characters")
    print(f"  Unique characters: {len(set(sample_text))}")
    
    # Experiment 1: Traditional (spike-only)
    print("\n" + "-" * 50)
    print("  Experiment 1: TRADITIONAL (Spike Count Only)")
    print("-" * 50)
    
    model_traditional = SNNLanguageModel(num_neurons=200)
    
    start = time.time()
    acc_trad, ppl_trad = train_and_evaluate(
        model_traditional, sample_text, use_hybrid=False, epochs=3
    )
    time_trad = time.time() - start
    
    print(f"    Accuracy: {acc_trad:.2f}%")
    print(f"    Perplexity: {ppl_trad:.2f}")
    print(f"    Time: {time_trad:.1f}s")
    
    # Experiment 2: Hybrid (spike + membrane)
    print("\n" + "-" * 50)
    print("  Experiment 2: HYBRID (Spike + Membrane Potential)")
    print("-" * 50)
    
    model_hybrid = SNNLanguageModel(num_neurons=200)
    
    start = time.time()
    acc_hybrid, ppl_hybrid = train_and_evaluate(
        model_hybrid, sample_text, use_hybrid=True, epochs=3
    )
    time_hybrid = time.time() - start
    
    print(f"    Accuracy: {acc_hybrid:.2f}%")
    print(f"    Perplexity: {ppl_hybrid:.2f}")
    print(f"    Time: {time_hybrid:.1f}s")
    
    # Comparison
    print("\n" + "=" * 70)
    print("   RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"""
    Metric          | Traditional | Hybrid      | Improvement
    ----------------|-------------|-------------|-------------
    Accuracy        | {acc_trad:>9.2f}%  | {acc_hybrid:>9.2f}%  | {acc_hybrid - acc_trad:>+.2f}%
    Perplexity      | {ppl_trad:>11.2f} | {ppl_hybrid:>11.2f} | {ppl_trad - ppl_hybrid:>+.2f}
    Time            | {time_trad:>9.1f}s  | {time_hybrid:>9.1f}s  | -
    """)
    
    if acc_hybrid > acc_trad:
        print("  🎉 HYBRID APPROACH WINS! Membrane potential adds useful information!")
    else:
        print("  🤔 No significant improvement. Try adjusting parameters.")
    
    # Save results
    with open("results/snn_lm_results.txt", "w", encoding="utf-8") as f:
        f.write("SNN Language Model Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Traditional (Spike only):\n")
        f.write(f"  Accuracy: {acc_trad:.2f}%\n")
        f.write(f"  Perplexity: {ppl_trad:.2f}\n")
        f.write(f"\nHybrid (Spike + Membrane):\n")
        f.write(f"  Accuracy: {acc_hybrid:.2f}%\n")
        f.write(f"  Perplexity: {ppl_hybrid:.2f}\n")
        f.write(f"\nImprovement: {acc_hybrid - acc_trad:+.2f}%\n")
    
    print("\n  Results saved to: results/snn_lm_results.txt")


if __name__ == "__main__":
    main()
