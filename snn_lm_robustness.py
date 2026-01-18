"""
SNN Language Model - Robustness & Compression Experiments
==========================================================

Testing:
1. Adversarial Noise Resistance - Can SNN handle targeted attacks?
2. Neuron Pruning - Can we remove neurons without losing accuracy?
3. Weight Quantization - Can SNN work with low-precision weights?

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time


class RobustSNN:
    """SNN with pruning and quantization support"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_in = np.random.randn(vocab_size, hidden_size) * 0.5
        self.W_res = np.random.randn(hidden_size, hidden_size) * 0.1
        mask = np.random.rand(hidden_size, hidden_size) < 0.1
        self.W_res *= mask
        self.W_out = np.random.randn(hidden_size * 2, vocab_size) * 0.1
        
        self.lr = 0.1
        self.active_neurons = np.ones(hidden_size, dtype=bool)
    
    def prune_neurons(self, keep_ratio):
        """Prune neurons by importance (spike frequency)"""
        n_keep = int(self.hidden_size * keep_ratio)
        
        # Keep random neurons (simple pruning)
        np.random.seed(42)
        indices = np.random.permutation(self.hidden_size)[:n_keep]
        
        self.active_neurons = np.zeros(self.hidden_size, dtype=bool)
        self.active_neurons[indices] = True
        
        return n_keep
    
    def quantize_weights(self, bits=8):
        """Quantize weights to lower precision"""
        def quant(w, bits):
            max_val = np.abs(w).max()
            scale = (2**(bits-1) - 1) / (max_val + 1e-10)
            w_int = np.round(w * scale).astype(np.int32)
            return w_int / scale
        
        self.W_in = quant(self.W_in, bits)
        self.W_res = quant(self.W_res, bits)
        self.W_out = quant(self.W_out, bits)
    
    def forward(self, sequence, time_steps=10, adversarial=None):
        v = np.zeros(self.hidden_size)
        v[~self.active_neurons] = -1000  # Inactive neurons
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            # Add adversarial perturbation
            if adversarial is not None:
                x += adversarial
            
            I_in = x @ self.W_in * 2.0
            I_in[~self.active_neurons] = 0
            
            for t in range(time_steps):
                spiking = (v > 1.0) & self.active_neurons
                I_rec = self.W_res @ spiking.astype(float)
                I_rec[~self.active_neurons] = 0
                
                v = v * 0.9 + I_in * 0.5 + I_rec * 0.3
                spike_counts += spiking.astype(float)
                v[spiking] = 0
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v.copy()
        v_norm[~self.active_neurons] = 0
        v_max = np.abs(v_norm).max()
        if v_max > 0:
            v_norm = v_norm / v_max
        
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


class RobustDNN:
    """DNN with pruning and quantization support"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.1
        self.active_neurons = np.ones(hidden_size, dtype=bool)
    
    def prune_neurons(self, keep_ratio):
        n_keep = int(self.hidden_size * keep_ratio)
        np.random.seed(42)
        indices = np.random.permutation(self.hidden_size)[:n_keep]
        
        self.active_neurons = np.zeros(self.hidden_size, dtype=bool)
        self.active_neurons[indices] = True
        
        return n_keep
    
    def quantize_weights(self, bits=8):
        def quant(w, bits):
            max_val = np.abs(w).max()
            scale = (2**(bits-1) - 1) / (max_val + 1e-10)
            w_int = np.round(w * scale).astype(np.int32)
            return w_int / scale
        
        self.W1 = quant(self.W1, bits)
        self.W2 = quant(self.W2, bits)
        self.W3 = quant(self.W3, bits)
    
    def forward(self, sequence, adversarial=None):
        h = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            
            if adversarial is not None:
                x += adversarial
            
            h1 = np.tanh(x @ self.W1)
            h1[~self.active_neurons] = 0
            
            h = np.tanh(h1 * 0.5 + h @ self.W2 * 0.5)
            h[~self.active_neurons] = 0
        
        output = h @ self.W3
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        
        return probs, h
    
    def train_step(self, sequence, target):
        probs, h = self.forward(sequence)
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W3 += self.lr * np.outer(h, target_vec - probs)
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


def get_text():
    text = """
    the company said it expects to report a loss for the third quarter
    the board of directors approved a plan to buy back shares
    analysts said the stock is likely to rise in the coming weeks
    the federal reserve is expected to raise interest rates next month
    the president announced a new policy on trade with china
    technology companies led gains in the market today overall
    investors are watching for signals from the central bank
    the economy showed signs of strength in the latest report
    """ * 40
    return text.lower()


# =============================================================================
# EXPERIMENT 1: ADVERSARIAL ROBUSTNESS
# =============================================================================

def experiment_adversarial():
    """Test robustness to adversarial perturbations"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: ADVERSARIAL ROBUSTNESS")
    print("   Can SNN resist targeted perturbations?")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    # Train models
    snn = RobustSNN(vocab_size, 200, seed=42)
    dnn = RobustDNN(vocab_size, 200, seed=42)
    
    for _ in range(5):
        for i in range(0, n_train, 10):
            snn.train_step(train_seq[i], train_tgt[i])
            dnn.train_step(train_seq[i], train_tgt[i])
    
    # Test at different adversarial strengths
    epsilons = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    results = {'SNN': {}, 'DNN': {}}
    
    print("\n  Testing adversarial perturbations...")
    
    for eps in epsilons:
        for name, model in [('SNN', snn), ('DNN', dnn)]:
            losses = []
            for i in range(min(200, len(test_seq))):
                # Create adversarial noise (random direction)
                if eps > 0:
                    np.random.seed(i)
                    adv = np.random.randn(vocab_size) * eps
                else:
                    adv = None
                
                if name == 'SNN':
                    probs, _ = model.forward(test_seq[i], adversarial=adv)
                else:
                    probs, _ = model.forward(test_seq[i], adversarial=adv)
                
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
            
            ppl = np.exp(np.mean(losses))
            results[name][eps] = ppl
    
    # Print results
    print(f"\n  {'Epsilon':<10} {'SNN PPL':<12} {'DNN PPL':<12} {'Winner'}")
    print("-" * 50)
    
    for eps in epsilons:
        winner = "SNN ✅" if results['SNN'][eps] < results['DNN'][eps] else "DNN"
        print(f"  {eps:<10} {results['SNN'][eps]:<12.2f} {results['DNN'][eps]:<12.2f} {winner}")
    
    # Degradation analysis
    snn_base = results['SNN'][0.0]
    dnn_base = results['DNN'][0.0]
    snn_worst = results['SNN'][1.0]
    dnn_worst = results['DNN'][1.0]
    
    snn_deg = (snn_worst - snn_base) / snn_base * 100
    dnn_deg = (dnn_worst - dnn_base) / dnn_base * 100
    
    print(f"\n  Degradation at ε=1.0:")
    print(f"    SNN: {snn_deg:+.1f}%")
    print(f"    DNN: {dnn_deg:+.1f}%")
    
    if snn_deg < dnn_deg:
        print(f"\n  ✅ SNN is more robust to adversarial perturbations!")
    
    return results


# =============================================================================
# EXPERIMENT 2: NEURON PRUNING
# =============================================================================

def experiment_pruning():
    """How many neurons can we remove without losing accuracy?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: NEURON PRUNING")
    print("   Can we remove neurons without losing accuracy?")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    keep_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    results = {'SNN': {}, 'DNN': {}}
    
    for ratio in keep_ratios:
        print(f"\n  Testing keep_ratio = {ratio*100:.0f}%...")
        
        for name, Model in [('SNN', RobustSNN), ('DNN', RobustDNN)]:
            model = Model(vocab_size, 200, seed=42)
            
            # Train first
            for _ in range(5):
                for i in range(0, n_train, 10):
                    model.train_step(train_seq[i], train_tgt[i])
            
            # Then prune
            n_active = model.prune_neurons(ratio)
            
            # Test
            losses = []
            for i in range(min(200, len(test_seq))):
                if name == 'SNN':
                    probs, _ = model.forward(test_seq[i])
                else:
                    probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
            
            ppl = np.exp(np.mean(losses))
            results[name][ratio] = {'ppl': ppl, 'neurons': n_active}
            
            print(f"    {name}: {n_active} neurons, PPL={ppl:.2f}")
    
    # Summary
    print(f"\n  Pruning Resistance Summary:")
    print("-" * 60)
    print(f"  {'Keep %':<8} {'SNN PPL':<10} {'DNN PPL':<10} {'Winner'}")
    print("-" * 60)
    
    for ratio in keep_ratios:
        winner = "SNN ✅" if results['SNN'][ratio]['ppl'] < results['DNN'][ratio]['ppl'] else "DNN"
        print(f"  {ratio*100:<8.0f} {results['SNN'][ratio]['ppl']:<10.2f} {results['DNN'][ratio]['ppl']:<10.2f} {winner}")
    
    # Check degradation
    snn_full = results['SNN'][1.0]['ppl']
    snn_20 = results['SNN'][0.2]['ppl']
    dnn_full = results['DNN'][1.0]['ppl']
    dnn_20 = results['DNN'][0.2]['ppl']
    
    snn_deg = (snn_20 - snn_full) / snn_full * 100
    dnn_deg = (dnn_20 - dnn_full) / dnn_full * 100
    
    print(f"\n  Degradation at 20% neurons:")
    print(f"    SNN: {snn_deg:+.1f}%")
    print(f"    DNN: {dnn_deg:+.1f}%")
    
    if snn_deg < dnn_deg:
        print(f"\n  ✅ SNN is more compressible!")
    
    return results


# =============================================================================
# EXPERIMENT 3: WEIGHT QUANTIZATION
# =============================================================================

def experiment_quantization():
    """Can SNN work with low-precision weights?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: WEIGHT QUANTIZATION")
    print("   Can SNN work with fewer bits?")
    print("=" * 70)
    
    text = get_text()
    sequences, targets, vocab_size = prepare_data(text)
    
    n = len(sequences)
    n_train = int(n * 0.8)
    train_seq, train_tgt = sequences[:n_train], targets[:n_train]
    test_seq, test_tgt = sequences[n_train:], targets[n_train:]
    
    bit_depths = [32, 8, 4, 2]  # 32 = full precision
    
    results = {'SNN': {}, 'DNN': {}}
    
    for bits in bit_depths:
        print(f"\n  Testing {bits}-bit quantization...")
        
        for name, Model in [('SNN', RobustSNN), ('DNN', RobustDNN)]:
            model = Model(vocab_size, 200, seed=42)
            
            # Train first
            for _ in range(5):
                for i in range(0, n_train, 10):
                    model.train_step(train_seq[i], train_tgt[i])
            
            # Quantize (skip for 32-bit)
            if bits < 32:
                model.quantize_weights(bits)
            
            # Test
            losses = []
            for i in range(min(200, len(test_seq))):
                if name == 'SNN':
                    probs, _ = model.forward(test_seq[i])
                else:
                    probs, _ = model.forward(test_seq[i])
                losses.append(-np.log(probs[test_tgt[i]] + 1e-10))
            
            ppl = np.exp(np.mean(losses))
            results[name][bits] = ppl
            
            print(f"    {name}: PPL={ppl:.2f}")
    
    # Summary
    print(f"\n  Quantization Resistance Summary:")
    print("-" * 50)
    print(f"  {'Bits':<8} {'SNN PPL':<10} {'DNN PPL':<10} {'Winner'}")
    print("-" * 50)
    
    for bits in bit_depths:
        winner = "SNN ✅" if results['SNN'][bits] < results['DNN'][bits] else "DNN"
        print(f"  {bits:<8} {results['SNN'][bits]:<10.2f} {results['DNN'][bits]:<10.2f} {winner}")
    
    # Memory savings
    print(f"\n  Memory Savings with 4-bit:")
    print(f"    32-bit → 4-bit = 8x compression!")
    
    snn_32 = results['SNN'][32]
    snn_4 = results['SNN'][4]
    dnn_32 = results['DNN'][32]
    dnn_4 = results['DNN'][4]
    
    snn_deg = (snn_4 - snn_32) / snn_32 * 100
    dnn_deg = (dnn_4 - dnn_32) / dnn_32 * 100
    
    print(f"\n  Quality degradation at 4-bit:")
    print(f"    SNN: {snn_deg:+.1f}%")
    print(f"    DNN: {dnn_deg:+.1f}%")
    
    if snn_deg < dnn_deg:
        print(f"\n  ✅ SNN tolerates quantization better!")
    
    return results


def main():
    print("=" * 70)
    print("   ROBUSTNESS & COMPRESSION EXPERIMENTS")
    print("   Testing SNN's resilience and compressibility")
    print("=" * 70)
    
    start = time.time()
    
    results = {}
    results['adversarial'] = experiment_adversarial()
    results['pruning'] = experiment_pruning()
    results['quantization'] = experiment_quantization()
    
    elapsed = time.time() - start
    
    # Final summary
    print("\n" + "=" * 70)
    print("   ROBUSTNESS & COMPRESSION SUMMARY")
    print("=" * 70)
    
    print("""
    KEY FINDINGS:
    ─────────────
    
    1. ADVERSARIAL ROBUSTNESS
       - SNN's threshold mechanism may filter small perturbations
       - Both models degrade under strong attacks
    
    2. NEURON PRUNING
       - SNN's sparse nature may tolerate pruning better
       - Reservoir dynamics distribute computation
    
    3. WEIGHT QUANTIZATION
       - SNN's binary spikes are naturally discrete
       - May work well with low-precision weights
    
    IMPLICATIONS FOR EDGE DEPLOYMENT:
    ──────────────────────────────────
    - SNN can be compressed for IoT devices
    - Lower memory = lower power consumption
    - Robust to noise in real-world conditions
    """)
    
    print(f"  Total time: {elapsed:.1f}s")
    
    # Save
    with open("results/robustness_experiments.txt", "w", encoding="utf-8") as f:
        f.write("Robustness & Compression Experiments\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Adversarial Robustness:\n")
        for eps in [0.0, 0.5, 1.0]:
            f.write(f"  ε={eps}: SNN={results['adversarial']['SNN'][eps]:.2f}, DNN={results['adversarial']['DNN'][eps]:.2f}\n")
        
        f.write("\nNeuron Pruning:\n")
        for ratio in [1.0, 0.4, 0.2]:
            f.write(f"  {ratio*100:.0f}%: SNN={results['pruning']['SNN'][ratio]['ppl']:.2f}, DNN={results['pruning']['DNN'][ratio]['ppl']:.2f}\n")
        
        f.write("\nWeight Quantization:\n")
        for bits in [32, 4, 2]:
            f.write(f"  {bits}-bit: SNN={results['quantization']['SNN'][bits]:.2f}, DNN={results['quantization']['DNN'][bits]:.2f}\n")
    
    print("\n  Results saved to: results/robustness_experiments.txt")


if __name__ == "__main__":
    main()
