"""
SNN Language Model - Parallel Large Scale Comparison
=====================================================

Using multiprocessing to run experiments faster.

Author: Hiroto Funasaki (roll)
Date: 2026-01-18
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import time


# ============================================
# Models (same as before, but pickable)
# ============================================

def create_snn(num_neurons, vocab_size, seed):
    """Create SNN model state (dict instead of class for pickle)"""
    np.random.seed(seed)
    
    W_res = np.random.randn(num_neurons, num_neurons) * 0.5
    rho = max(abs(np.linalg.eigvals(W_res)))
    W_res *= 1.4 / rho
    mask = np.random.rand(num_neurons, num_neurons) < 0.1
    W_res *= mask
    
    return {
        'type': 'SNN',
        'name': f'SNN-{num_neurons}',
        'num_neurons': num_neurons,
        'vocab_size': vocab_size,
        'W_res': W_res,
        'W_in': np.random.randn(num_neurons, vocab_size) * 0.3,
        'W_out': np.zeros((vocab_size, num_neurons)),
        'W_out_m': np.zeros((vocab_size, num_neurons)),
        'v': np.full(num_neurons, -65.0),
        'fire_rate': np.zeros(num_neurons),
        'alpha': 0.01,
        'ops': 0
    }


def create_dnn(hidden_size, vocab_size, seed):
    """Create DNN model state"""
    np.random.seed(seed)
    
    return {
        'type': 'DNN',
        'name': f'DNN-{hidden_size}',
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'W1': np.random.randn(hidden_size, vocab_size) * 0.1,
        'W2': np.random.randn(hidden_size, hidden_size) * 0.1,
        'W3': np.random.randn(vocab_size, hidden_size) * 0.1,
        'h': np.zeros(hidden_size),
        'alpha': 0.005,
        'ops': 0
    }


def create_lstm(hidden_size, vocab_size, seed):
    """Create LSTM model state"""
    np.random.seed(seed)
    combined = hidden_size + vocab_size
    
    return {
        'type': 'LSTM',
        'name': f'LSTM-{hidden_size}',
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'combined': combined,
        'Wf': np.random.randn(hidden_size, combined) * 0.1,
        'Wi': np.random.randn(hidden_size, combined) * 0.1,
        'Wc': np.random.randn(hidden_size, combined) * 0.1,
        'Wo': np.random.randn(hidden_size, combined) * 0.1,
        'Wy': np.random.randn(vocab_size, hidden_size) * 0.1,
        'h': np.zeros(hidden_size),
        'c': np.zeros(hidden_size),
        'alpha': 0.005,
        'ops': 0
    }


def snn_step(model, char_idx):
    """SNN forward step"""
    x = np.zeros(model['vocab_size'])
    x[char_idx] = 1.0
    
    I_in = model['W_in'] @ x
    I_rec = model['W_res'] @ model['fire_rate']
    I_total = I_in + I_rec + np.random.normal(0, 0.1, model['num_neurons']) + 25.0
    
    # LIF dynamics
    spikes = np.zeros(model['num_neurons'])
    potentials = np.zeros(model['num_neurons'])
    
    for i in range(model['num_neurons']):
        dv = (-(model['v'][i] + 65.0) + I_total[i]) / 20.0 * 0.5
        model['v'][i] += dv
        
        if model['v'][i] >= -50.0:
            model['v'][i] = -70.0
            spikes[i] = 1.0
        
        potentials[i] = (model['v'][i] + 70.0) / 20.0
    
    model['fire_rate'] = 0.8 * model['fire_rate'] + 0.2 * spikes
    model['ops'] += int(np.sum(spikes > 0) * model['num_neurons'] * 2)
    
    return spikes, potentials


def dnn_step(model, char_idx):
    """DNN forward step"""
    x = np.zeros(model['vocab_size'])
    x[char_idx] = 1.0
    
    h1 = np.tanh(model['W1'] @ x)
    h2 = np.tanh(model['W2'] @ model['h'] + h1)
    model['h'] = h2
    
    model['ops'] += model['hidden_size'] * (model['vocab_size'] + model['hidden_size']) * 2
    
    return h2


def lstm_step(model, char_idx):
    """LSTM forward step"""
    x = np.zeros(model['vocab_size'])
    x[char_idx] = 1.0
    
    hx = np.concatenate([model['h'], x])
    
    f = 1 / (1 + np.exp(-np.clip(model['Wf'] @ hx, -500, 500)))
    i = 1 / (1 + np.exp(-np.clip(model['Wi'] @ hx, -500, 500)))
    c_tilde = np.tanh(model['Wc'] @ hx)
    o = 1 / (1 + np.exp(-np.clip(model['Wo'] @ hx, -500, 500)))
    
    model['c'] = f * model['c'] + i * c_tilde
    model['h'] = o * np.tanh(model['c'])
    
    model['ops'] += 4 * model['hidden_size'] * model['combined'] * 2
    
    return model['h']


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_model_task(args):
    """Run single training task (for parallel execution)"""
    model_type, size, text, epochs, noise, seed = args
    
    np.random.seed(seed)
    vocab_size = 128
    
    # Create model
    if model_type == 'SNN':
        model = create_snn(size, vocab_size, seed)
    elif model_type == 'DNN':
        model = create_dnn(size, vocab_size, seed)
    else:
        model = create_lstm(size, vocab_size, seed)
    
    correct = 0
    total = 0
    loss_sum = 0
    
    for epoch in range(epochs):
        # Reset state
        if model_type == 'SNN':
            model['v'] = np.full(model['num_neurons'], -65.0)
            model['fire_rate'] = np.zeros(model['num_neurons'])
        elif model_type == 'DNN':
            model['h'] = np.zeros(model['hidden_size'])
        else:
            model['h'] = np.zeros(model['hidden_size'])
            model['c'] = np.zeros(model['hidden_size'])
        
        for idx in range(len(text) - 1):
            char_idx = ord(text[idx]) % vocab_size
            next_idx = ord(text[idx + 1]) % vocab_size
            
            # Add noise
            if noise > 0 and np.random.random() < noise:
                char_idx = np.random.randint(0, vocab_size)
            
            # Forward
            if model_type == 'SNN':
                spikes, potentials = snn_step(model, char_idx)
                logits = model['W_out'] @ spikes + model['W_out_m'] @ potentials
                pred = softmax(logits)
                
                # Train
                target = np.zeros(vocab_size)
                target[next_idx] = 1.0
                error = target - pred
                model['W_out'] += model['alpha'] * np.outer(error, spikes)
                model['W_out_m'] += model['alpha'] * np.outer(error, potentials)
            else:
                if model_type == 'DNN':
                    hidden = dnn_step(model, char_idx)
                else:
                    hidden = lstm_step(model, char_idx)
                
                logits = model['Wy' if model_type == 'LSTM' else 'W3'] @ hidden
                pred = softmax(logits)
                
                target = np.zeros(vocab_size)
                target[next_idx] = 1.0
                error = target - pred
                if model_type == 'LSTM':
                    model['Wy'] += model['alpha'] * np.outer(error, hidden)
                else:
                    model['W3'] += model['alpha'] * np.outer(error, hidden)
            
            if np.argmax(pred) == next_idx:
                correct += 1
            total += 1
            loss_sum += -np.log(pred[next_idx] + 1e-10)
    
    accuracy = correct / total * 100
    perplexity = np.exp(loss_sum / total)
    
    return (model['name'], accuracy, perplexity, model['ops'])


def main():
    print("=" * 70)
    print("   SNN LANGUAGE MODEL - PARALLEL LARGE SCALE COMPARISON")
    print(f"   Using {cpu_count()} CPU cores")
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
    Deep learning has enabled machines to understand human language.
    The transformer architecture has become the dominant approach.
    However, transformers are computationally expensive.
    Spiking neural networks offer an alternative approach.
    They are more energy efficient and can run on neuromorphic hardware.
    """ * 25
    
    sample_text = ''.join(c for c in sample_text if 32 <= ord(c) < 128)
    print(f"\n  Training text: {len(sample_text)} characters")
    
    start_total = time.time()
    
    # ================================
    # Experiment 1: Scale Comparison (Parallel)
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: Scale Comparison (Parallel)")
    print("=" * 70)
    
    tasks = []
    for size in [300, 500]:
        tasks.append(('SNN', size, sample_text, 2, 0.0, 42))
        tasks.append(('DNN', size, sample_text, 2, 0.0, 42))
    for size in [200, 300]:
        tasks.append(('LSTM', size, sample_text, 2, 0.0, 42))
    
    start = time.time()
    with Pool(cpu_count()) as pool:
        results = pool.map(train_model_task, tasks)
    elapsed = time.time() - start
    
    print(f"\n  Completed {len(tasks)} models in {elapsed:.1f}s (parallel)")
    print(f"\n  {'Model':<12} | {'Accuracy':<10} | {'PPL':<10} | {'Ops (M)':<10}")
    print("-" * 55)
    
    results_dict = {}
    for name, acc, ppl, ops in results:
        results_dict[name] = (acc, ppl, ops)
        print(f"  {name:<12} | {acc:>8.2f}% | {ppl:>8.2f} | {ops/1e6:>8.2f}")
    
    # ================================
    # Experiment 2: Noise Robustness (Parallel)
    # ================================
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Noise Robustness (Parallel)")
    print("=" * 70)
    
    noise_tasks = []
    for noise in [0.0, 0.1, 0.2, 0.3]:
        noise_tasks.append(('SNN', 500, sample_text, 2, noise, 42))
        noise_tasks.append(('DNN', 500, sample_text, 2, noise, 42))
        noise_tasks.append(('LSTM', 300, sample_text, 2, noise, 42))
    
    start = time.time()
    with Pool(cpu_count()) as pool:
        noise_results = pool.map(train_model_task, noise_tasks)
    elapsed = time.time() - start
    
    print(f"\n  Completed noise tests in {elapsed:.1f}s (parallel)")
    
    # Organize results
    noise_by_model = {'SNN-500': [], 'DNN-500': [], 'LSTM-300': []}
    for i, (name, acc, ppl, ops) in enumerate(noise_results):
        noise_level = [0.0, 0.1, 0.2, 0.3][i // 3]
        if name in noise_by_model:
            noise_by_model[name].append((noise_level, acc))
    
    print(f"\n  {'Model':<12} | {'0%':<8} | {'10%':<8} | {'20%':<8} | {'30%':<8} | Degrade")
    print("-" * 70)
    
    for name, data in noise_by_model.items():
        print(f"  {name:<12}", end=" |")
        for noise, acc in data:
            print(f" {acc:>6.2f}%", end=" |")
        if len(data) >= 2:
            degradation = data[0][1] - data[-1][1]
            symbol = "✅" if degradation < 2.0 else "⚠️"
            print(f" {degradation:>+.2f}% {symbol}")
        else:
            print()
    
    # ================================
    # Summary
    # ================================
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    
    # Energy efficiency
    snn_eff = results_dict['SNN-300'][0] / (results_dict['SNN-300'][2] / 1e6)
    dnn_eff = results_dict['DNN-300'][0] / (results_dict['DNN-300'][2] / 1e6)
    ratio = snn_eff / dnn_eff if dnn_eff > 0 else 0
    
    print(f"\n  Energy Efficiency:")
    print(f"    SNN-300: {snn_eff:.4f} acc/MOps")
    print(f"    DNN-300: {dnn_eff:.4f} acc/MOps")
    print(f"    → SNN is {ratio:.1f}x more efficient!")
    
    # Noise robustness
    snn_deg = noise_by_model['SNN-500'][0][1] - noise_by_model['SNN-500'][-1][1]
    dnn_deg = noise_by_model['DNN-500'][0][1] - noise_by_model['DNN-500'][-1][1]
    
    print(f"\n  Noise Robustness (0% → 30%):")
    print(f"    SNN-500: {snn_deg:+.2f}%")
    print(f"    DNN-500: {dnn_deg:+.2f}%")
    
    if snn_deg < dnn_deg:
        print(f"    → SNN is MORE ROBUST by {dnn_deg - snn_deg:.2f}%!")
    
    total_time = time.time() - start_total
    print(f"\n  Total time: {total_time:.1f}s")
    
    # Save
    with open("results/parallel_comparison.txt", "w", encoding="utf-8") as f:
        f.write("SNN Language Model - Parallel Comparison Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Text: {len(sample_text)} chars\n")
        f.write(f"CPUs: {cpu_count()}\n\n")
        
        f.write("Accuracy:\n")
        for name, (acc, ppl, ops) in results_dict.items():
            f.write(f"  {name}: {acc:.2f}%\n")
        
        f.write(f"\nEnergy Efficiency: SNN is {ratio:.1f}x better\n")
        f.write(f"Noise Robustness: SNN degrades {snn_deg:.2f}% vs DNN {dnn_deg:.2f}%\n")
    
    print("\n  Results saved to: results/parallel_comparison.txt")


if __name__ == "__main__":
    main()
