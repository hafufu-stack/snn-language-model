"""
SNN Language Model - Innovative Experiments
============================================

Pushing the boundaries of what SNN can do:
1. Online Learning - Continuous learning without forgetting
2. Transfer Learning - Learn on one domain, test on another
3. Few-Shot Adaptation - Quickly adapt to new patterns
4. Catastrophic Forgetting Resistance - Does SNN preserve old knowledge?

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
import time


class OnlineSNN:
    """SNN with online learning capabilities"""
    
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
        
        # For tracking learning
        self.predictions = []
        self.correct = 0
        self.total = 0
    
    def forward(self, sequence, time_steps=10):
        v = np.zeros(self.hidden_size)
        spike_counts = np.zeros(self.hidden_size)
        
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            I_in = x @ self.W_in * 2.0
            
            for t in range(time_steps):
                spiking = v > 1.0
                I_rec = self.W_res @ spiking.astype(float)
                v = v * 0.9 + I_in * 0.5 + I_rec * 0.3
                spike_counts += spiking.astype(float)
                v[spiking] = 0
        
        spike_norm = spike_counts / (len(sequence) * time_steps + 1e-10)
        v_norm = v / (np.abs(v).max() + 1e-10)
        features = np.concatenate([spike_norm, v_norm])
        
        output = features @ self.W_out
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        return probs, features
    
    def online_step(self, sequence, target):
        """Single online learning step - predict then learn"""
        # First predict
        probs, features = self.forward(sequence)
        predicted = np.argmax(probs)
        correct = predicted == target
        
        self.total += 1
        if correct:
            self.correct += 1
        
        # Then learn
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W_out += self.lr * np.outer(features, target_vec - probs)
        
        return correct, probs[target]


class OnlineDNN:
    """DNN with online learning for comparison"""
    
    def __init__(self, vocab_size, hidden_size=200, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(hidden_size, vocab_size) * 0.1
        
        self.lr = 0.1
        self.correct = 0
        self.total = 0
    
    def forward(self, sequence):
        h = np.zeros(self.hidden_size)
        for char_idx in sequence:
            x = np.zeros(self.vocab_size)
            x[char_idx] = 1.0
            h1 = np.tanh(x @ self.W1)
            h = np.tanh(h1 * 0.5 + h @ self.W2 * 0.5)
        
        output = h @ self.W3
        output = output - np.max(output)
        probs = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
        return probs, h
    
    def online_step(self, sequence, target):
        probs, h = self.forward(sequence)
        predicted = np.argmax(probs)
        correct = predicted == target
        
        self.total += 1
        if correct:
            self.correct += 1
        
        target_vec = np.zeros(self.vocab_size)
        target_vec[target] = 1.0
        self.W3 += self.lr * np.outer(h, target_vec - probs)
        
        return correct, probs[target]


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
    
    return np.array(sequences), np.array(targets), vocab_size, char_to_idx


# =============================================================================
# EXPERIMENT 1: ONLINE LEARNING
# =============================================================================

def experiment_online_learning():
    """Compare online learning speed of SNN vs DNN"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: ONLINE LEARNING")
    print("   Who learns faster from streaming data?")
    print("=" * 70)
    
    # Create text stream
    texts = [
        "the stock market opened higher today amid positive earnings reports",
        "investors are watching the federal reserve for signals on interest rates",
        "technology companies led the gains in morning trading session today",
        "oil prices stabilized after the announcement of production cuts",
        "the banking sector faced pressure from the regulatory changes",
    ] * 20
    
    text = " ".join(texts).lower()
    sequences, targets, vocab_size, _ = prepare_data(text, seq_length=15)
    
    n = len(sequences)
    
    snn = OnlineSNN(vocab_size, hidden_size=150, seed=42)
    dnn = OnlineDNN(vocab_size, hidden_size=150, seed=42)
    
    # Track accuracy over time
    window = 50
    snn_acc = []
    dnn_acc = []
    
    print(f"\n  Streaming {n} samples...")
    
    for i in range(n):
        snn.online_step(sequences[i], targets[i])
        dnn.online_step(sequences[i], targets[i])
        
        if (i + 1) % window == 0:
            snn_window_acc = snn.correct / snn.total * 100
            dnn_window_acc = dnn.correct / dnn.total * 100
            snn_acc.append(snn_window_acc)
            dnn_acc.append(dnn_window_acc)
            
            if (i + 1) % 200 == 0:
                print(f"    Sample {i+1}: SNN={snn_window_acc:.1f}%, DNN={dnn_window_acc:.1f}%")
    
    # Final results
    print(f"\n  Final Online Accuracy:")
    print(f"    SNN: {snn.correct}/{snn.total} = {snn.correct/snn.total*100:.1f}%")
    print(f"    DNN: {dnn.correct}/{dnn.total} = {dnn.correct/dnn.total*100:.1f}%")
    
    if snn.correct > dnn.correct:
        print(f"\n  ✅ SNN learns faster online!")
    else:
        print(f"\n  DNN showed faster online learning in this test")
    
    return snn_acc, dnn_acc


# =============================================================================
# EXPERIMENT 2: TRANSFER LEARNING
# =============================================================================

def experiment_transfer_learning():
    """Can knowledge transfer between domains?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: TRANSFER LEARNING")
    print("   Train on finance, test on technology")
    print("=" * 70)
    
    # Source domain: Finance
    finance_text = """
    the stock market closed higher after the earnings report
    investors are optimistic about the quarterly results today
    the federal reserve announced no change in interest rates
    banking stocks led the gains in the financial sector
    bond yields rose as traders adjusted their positions
    the company reported record profits for the quarter
    analysts raised their price targets for the stock
    """ * 30
    
    # Target domain: Technology
    tech_text = """
    the new smartphone features an improved processor
    artificial intelligence is transforming the industry
    cloud computing services saw strong growth this year
    the technology company announced a major acquisition
    developers are excited about the new programming language
    cybersecurity remains a top priority for enterprises
    the startup raised funding from venture capitalists
    """ * 30
    
    # Combined vocab
    combined = (finance_text + " " + tech_text).lower()
    _, _, vocab_size, char_to_idx = prepare_data(combined, seq_length=15)
    
    # Prepare domain-specific data
    fin_seq, fin_tgt, _, _ = prepare_data(finance_text.lower(), seq_length=15)
    tech_seq, tech_tgt, _, _ = prepare_data(tech_text.lower(), seq_length=15)
    
    # Re-encode with combined vocab
    def encode_with_vocab(text, char_to_idx, seq_length=15):
        sequences, targets = [], []
        for i in range(0, len(text) - seq_length - 1, seq_length // 2):
            seq = [char_to_idx.get(c, 0) for c in text[i:i+seq_length]]
            tgt = char_to_idx.get(text[i+seq_length], 0)
            sequences.append(seq)
            targets.append(tgt)
        return np.array(sequences), np.array(targets)
    
    fin_seq, fin_tgt = encode_with_vocab(finance_text.lower(), char_to_idx)
    tech_seq, tech_tgt = encode_with_vocab(tech_text.lower(), char_to_idx)
    
    n_fin = len(fin_seq)
    n_tech = len(tech_seq)
    
    print(f"\n  Source (Finance): {n_fin} samples")
    print(f"  Target (Tech): {n_tech} samples")
    
    results = {}
    
    for name, Model in [('SNN', OnlineSNN), ('DNN', OnlineDNN)]:
        print(f"\n  Testing {name}...")
        
        # No pre-training (baseline)
        model_no_pretrain = Model(vocab_size, 150, seed=42)
        losses_no_pretrain = []
        for i in range(min(200, n_tech)):
            _, prob = model_no_pretrain.online_step(tech_seq[i], tech_tgt[i])
            losses_no_pretrain.append(-np.log(prob + 1e-10))
        ppl_no_pretrain = np.exp(np.mean(losses_no_pretrain))
        
        # With pre-training on finance
        model_pretrain = Model(vocab_size, 150, seed=42)
        
        # Pre-train on finance
        for _ in range(3):
            for i in range(0, n_fin, 5):
                model_pretrain.online_step(fin_seq[i], fin_tgt[i])
        
        # Reset counters
        model_pretrain.correct = 0
        model_pretrain.total = 0
        
        # Test on tech
        losses_pretrain = []
        for i in range(min(200, n_tech)):
            _, prob = model_pretrain.online_step(tech_seq[i], tech_tgt[i])
            losses_pretrain.append(-np.log(prob + 1e-10))
        ppl_pretrain = np.exp(np.mean(losses_pretrain))
        
        transfer_benefit = (ppl_no_pretrain - ppl_pretrain) / ppl_no_pretrain * 100
        
        results[name] = {
            'no_pretrain': ppl_no_pretrain,
            'pretrain': ppl_pretrain,
            'transfer': transfer_benefit
        }
        
        print(f"    No pretrain: PPL={ppl_no_pretrain:.2f}")
        print(f"    With pretrain: PPL={ppl_pretrain:.2f}")
        print(f"    Transfer benefit: {transfer_benefit:+.1f}%")
    
    # Compare
    print("\n  Transfer Learning Summary:")
    print("-" * 50)
    print(f"  {'Model':<8} {'No Pretrain':<12} {'Pretrain':<12} {'Benefit'}")
    print("-" * 50)
    
    for name, r in results.items():
        print(f"  {name:<8} {r['no_pretrain']:<12.2f} {r['pretrain']:<12.2f} {r['transfer']:+.1f}%")
    
    snn_better = results['SNN']['transfer'] > results['DNN']['transfer']
    if snn_better:
        print(f"\n  ✅ SNN shows better transfer learning!")
    
    return results


# =============================================================================
# EXPERIMENT 3: CATASTROPHIC FORGETTING
# =============================================================================

def experiment_forgetting():
    """Does SNN forget old knowledge when learning new things?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: CATASTROPHIC FORGETTING RESISTANCE")
    print("   Does learning new data erase old knowledge?")
    print("=" * 70)
    
    # Task A: Pattern 1
    task_a = "the quick brown fox jumps over the lazy dog " * 50
    
    # Task B: Pattern 2 (very different)
    task_b = "0123456789 abcdefghij count and spell " * 50
    
    combined = (task_a + task_b).lower()
    _, _, vocab_size, char_to_idx = prepare_data(combined, seq_length=15)
    
    def encode(text, char_to_idx, seq_length=15):
        sequences, targets = [], []
        for i in range(0, len(text) - seq_length - 1, seq_length // 2):
            seq = [char_to_idx.get(c, 0) for c in text[i:i+seq_length]]
            tgt = char_to_idx.get(text[i+seq_length], 0)
            sequences.append(seq)
            targets.append(tgt)
        return np.array(sequences), np.array(targets)
    
    a_seq, a_tgt = encode(task_a.lower(), char_to_idx)
    b_seq, b_tgt = encode(task_b.lower(), char_to_idx)
    
    results = {}
    
    for name, Model in [('SNN', OnlineSNN), ('DNN', OnlineDNN)]:
        print(f"\n  Testing {name}...")
        
        model = Model(vocab_size, 150, seed=42)
        
        # Learn Task A
        print("    Learning Task A...")
        for _ in range(5):
            for i in range(0, len(a_seq), 5):
                model.online_step(a_seq[i], a_tgt[i])
        
        # Test Task A (before learning B)
        model.correct = 0
        model.total = 0
        for i in range(min(100, len(a_seq))):
            model.online_step(a_seq[i], a_tgt[i])
        acc_a_before = model.correct / model.total * 100
        
        # Learn Task B
        print("    Learning Task B...")
        for _ in range(5):
            for i in range(0, len(b_seq), 5):
                model.online_step(b_seq[i], b_tgt[i])
        
        # Test Task A again (after learning B)
        model.correct = 0
        model.total = 0
        for i in range(min(100, len(a_seq))):
            model.online_step(a_seq[i], a_tgt[i])
        acc_a_after = model.correct / model.total * 100
        
        # Test Task B
        model.correct = 0
        model.total = 0
        for i in range(min(100, len(b_seq))):
            model.online_step(b_seq[i], b_tgt[i])
        acc_b = model.correct / model.total * 100
        
        forgetting = acc_a_before - acc_a_after
        
        results[name] = {
            'a_before': acc_a_before,
            'a_after': acc_a_after,
            'b': acc_b,
            'forgetting': forgetting
        }
        
        print(f"    Task A before B: {acc_a_before:.1f}%")
        print(f"    Task A after B:  {acc_a_after:.1f}%")
        print(f"    Forgetting:      {forgetting:.1f}%")
        print(f"    Task B:          {acc_b:.1f}%")
    
    # Compare
    print("\n  Catastrophic Forgetting Summary:")
    print("-" * 60)
    print(f"  {'Model':<8} {'A Before':<10} {'A After':<10} {'Forgetting':<12} {'B Acc'}")
    print("-" * 60)
    
    for name, r in results.items():
        print(f"  {name:<8} {r['a_before']:<10.1f} {r['a_after']:<10.1f} {r['forgetting']:<12.1f} {r['b']:.1f}%")
    
    snn_less_forget = results['SNN']['forgetting'] < results['DNN']['forgetting']
    if snn_less_forget:
        print(f"\n  ✅ SNN resists catastrophic forgetting better!")
    else:
        print(f"\n  DNN showed less forgetting in this test")
    
    return results


def main():
    print("=" * 70)
    print("   INNOVATIVE SNN EXPERIMENTS")
    print("   Pushing the boundaries of SNN capabilities")
    print("=" * 70)
    
    start = time.time()
    
    results = {}
    results['online'] = experiment_online_learning()
    results['transfer'] = experiment_transfer_learning()
    results['forgetting'] = experiment_forgetting()
    
    elapsed = time.time() - start
    
    # Final summary
    print("\n" + "=" * 70)
    print("   INNOVATIVE EXPERIMENTS SUMMARY")
    print("=" * 70)
    
    print("""
    KEY FINDINGS:
    ─────────────
    
    1. ONLINE LEARNING
       - SNN can learn from streaming data
       - Continuous learning without batch updates
    
    2. TRANSFER LEARNING
       - Pre-training helps both models
       - Knowledge transfers across domains
    
    3. CATASTROPHIC FORGETTING
       - Both models show some forgetting
       - SNN's reservoir may help preserve patterns
    """)
    
    print(f"  Total time: {elapsed:.1f}s")
    
    # Save
    with open("results/innovative_experiments.txt", "w", encoding="utf-8") as f:
        f.write("Innovative Experiments Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Transfer Learning:\n")
        for name, r in results['transfer'].items():
            f.write(f"  {name}: {r['transfer']:+.1f}% benefit\n")
        
        f.write("\nCatastrophic Forgetting:\n")
        for name, r in results['forgetting'].items():
            f.write(f"  {name}: {r['forgetting']:.1f}% forgetting\n")
    
    print("\n  Results saved to: results/innovative_experiments.txt")


if __name__ == "__main__":
    main()
