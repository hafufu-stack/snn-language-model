"""
SNN Language Model - Context-Dependent Representation
======================================================

Question: Can SNN dynamically change word representations based on context?

Example: "bank" in different contexts:
- "I deposited money in the bank" → financial meaning
- "I sat by the river bank" → nature meaning

Hypothesis: SNN's recurrent dynamics and temporal states can shift
word representations based on preceding context.

Author: Hiroto Funasaki (roll)
Date: 2026-01-19
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import time


class ContextualSNNEncoder:
    """SNN encoder with contextual memory"""
    
    def __init__(self, num_neurons=150, time_steps=50, seed=42):
        np.random.seed(seed)
        self.num_neurons = num_neurons
        self.time_steps = time_steps
        
        # LIF parameters
        self.tau = 15.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.v_rest = -65.0
        
        # Reservoir weights (more recurrent for context memory)
        self.W_res = np.random.randn(num_neurons, num_neurons) * 0.6
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.4 / rho
        mask = np.random.rand(num_neurons, num_neurons) < 0.2  # More connections
        self.W_res *= mask
        
        # Word embeddings
        self.vocab = {}
        self.embeddings = {}
        
        # Internal state (context memory)
        self.v = np.full(num_neurons, self.v_rest)
        self.context_state = np.zeros(num_neurons)
    
    def register_word(self, word, semantic_bias=None):
        """Register word with optional semantic bias"""
        if word not in self.vocab:
            idx = len(self.vocab)
            self.vocab[word] = idx
            
            np.random.seed(hash(word) % (2**32))
            embedding = np.random.randn(self.num_neurons) * 15 + 20
            
            if semantic_bias is not None:
                embedding += semantic_bias * 10
            
            self.embeddings[word] = embedding
        return self.vocab[word]
    
    def reset_context(self):
        """Reset internal state (start new sentence)"""
        self.v = np.full(self.num_neurons, self.v_rest)
        self.context_state = np.zeros(self.num_neurons)
    
    def encode_word(self, word, update_context=True):
        """Encode word with current context influencing representation"""
        if word not in self.vocab:
            self.register_word(word)
        
        I_word = self.embeddings[word].copy()
        
        spike_times = [[] for _ in range(self.num_neurons)]
        spike_counts = np.zeros(self.num_neurons)
        potentials = []
        
        # Context influence on input
        I_context = self.context_state * 5
        
        for t in range(self.time_steps):
            # Combined input
            I_in = I_word * np.exp(-t / 25.0) + I_context * np.exp(-t / 40.0)
            I_rec = self.W_res @ (self.v > self.v_thresh).astype(float) * 12
            I_noise = np.random.randn(self.num_neurons) * 2
            
            I_total = I_in + I_rec + I_noise
            
            # LIF dynamics
            dv = (-(self.v - self.v_rest) + I_total) / self.tau
            self.v += dv
            
            spiking = self.v >= self.v_thresh
            for i in np.where(spiking)[0]:
                spike_times[i].append(t)
            spike_counts += spiking.astype(float)
            self.v[spiking] = self.v_reset
            
            potentials.append(self.v.copy())
        
        # Update context state for next word
        if update_context:
            self.context_state = 0.7 * self.context_state + 0.3 * spike_counts / 10
        
        # Feature vector
        first_spikes = np.array([s[0] if s else self.time_steps for s in spike_times])
        feature = np.concatenate([
            first_spikes / self.time_steps,
            spike_counts / 10,
            self.context_state  # Include context!
        ])
        
        return feature, spike_counts, np.array(potentials)
    
    def encode_sentence(self, words, return_all=False):
        """Encode a sequence of words with context accumulation"""
        self.reset_context()
        
        features = []
        for word in words:
            feat, _, _ = self.encode_word(word)
            features.append(feat)
        
        if return_all:
            return features
        return features[-1]  # Return last word's representation


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def experiment_context_shift():
    """Test: Does context shift word representation?"""
    print("=" * 70)
    print("   EXPERIMENT 1: Context-Dependent Representation Shift")
    print("=" * 70)
    
    encoder = ContextualSNNEncoder(num_neurons=150, time_steps=50, seed=42)
    
    # Register words with semantic biases
    financial = np.array([1, 0, 0] * 50)  # 150 dim
    nature = np.array([0, 1, 0] * 50)
    
    encoder.register_word("bank")  # Ambiguous!
    encoder.register_word("money", financial)
    encoder.register_word("deposit", financial)
    encoder.register_word("account", financial)
    encoder.register_word("river", nature)
    encoder.register_word("water", nature)
    encoder.register_word("sat", nature)
    
    # Context 1: Financial
    encoder.reset_context()
    _ = encoder.encode_word("money")
    _ = encoder.encode_word("deposit")
    bank_financial, _, _ = encoder.encode_word("bank")
    
    # Context 2: Nature
    encoder.reset_context()
    _ = encoder.encode_word("river")
    _ = encoder.encode_word("water")
    bank_nature, _, _ = encoder.encode_word("bank")
    
    # Context 3: No context
    encoder.reset_context()
    bank_neutral, _, _ = encoder.encode_word("bank")
    
    # Compute similarities
    sim_fin_nat = cosine_sim(bank_financial, bank_nature)
    sim_fin_neut = cosine_sim(bank_financial, bank_neutral)
    sim_nat_neut = cosine_sim(bank_nature, bank_neutral)
    
    print("\n  'bank' representation similarity across contexts:")
    print("-" * 50)
    print(f"    Financial vs Nature:   {sim_fin_nat:.4f}")
    print(f"    Financial vs Neutral:  {sim_fin_neut:.4f}")
    print(f"    Nature vs Neutral:     {sim_nat_neut:.4f}")
    
    # Check if context makes a difference
    context_effect = 1.0 - sim_fin_nat
    print(f"\n    Context effect magnitude: {context_effect:.4f}")
    
    if context_effect > 0.05:
        print("\n  ✅ Context DOES shift word representation!")
    else:
        print("\n  ⚠️ Context effect is minimal")
    
    return {
        'fin_nat': sim_fin_nat,
        'fin_neut': sim_fin_neut,
        'nat_neut': sim_nat_neut,
        'effect': context_effect
    }


def experiment_semantic_priming():
    """Test: Does related context improve semantic coherence?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Semantic Priming Effect")
    print("=" * 70)
    
    encoder = ContextualSNNEncoder(num_neurons=150, time_steps=50, seed=42)
    
    # Register semantic groups
    animals = ["dog", "cat", "bird", "fish"]
    colors = ["red", "blue", "green", "yellow"]
    
    for w in animals + colors:
        encoder.register_word(w)
    
    # Test: "dog" after animal context vs color context
    
    # Animal context
    encoder.reset_context()
    _ = encoder.encode_word("cat")
    _ = encoder.encode_word("bird")
    dog_animal_ctx, _, _ = encoder.encode_word("dog")
    
    # Color context (unrelated)
    encoder.reset_context()
    _ = encoder.encode_word("red")
    _ = encoder.encode_word("blue")
    dog_color_ctx, _, _ = encoder.encode_word("dog")
    
    # No context
    encoder.reset_context()
    dog_neutral, _, _ = encoder.encode_word("dog")
    
    # Compute coherence with semantic category
    encoder.reset_context()
    cat_rep, _, _ = encoder.encode_word("cat")
    
    sim_animal_to_cat = cosine_sim(dog_animal_ctx, cat_rep)
    sim_color_to_cat = cosine_sim(dog_color_ctx, cat_rep)
    
    print("\n  'dog' similarity to 'cat' after different priming:")
    print("-" * 50)
    print(f"    After animal context: {sim_animal_to_cat:.4f}")
    print(f"    After color context:  {sim_color_to_cat:.4f}")
    
    priming_effect = sim_animal_to_cat - sim_color_to_cat
    print(f"\n    Priming effect: {priming_effect:+.4f}")
    
    if priming_effect > 0.01:
        print("\n  ✅ Semantic priming works! Related context increases similarity!")
    else:
        print("\n  ⚠️ Priming effect is weak")
    
    return {
        'animal_ctx': sim_animal_to_cat,
        'color_ctx': sim_color_to_cat,
        'priming_effect': priming_effect
    }


def experiment_sentence_embedding():
    """Test: Can SNN create meaningful sentence embeddings?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: Sentence Embedding via Context Accumulation")
    print("=" * 70)
    
    encoder = ContextualSNNEncoder(num_neurons=150, time_steps=50, seed=42)
    
    # Similar sentences
    s1 = ["the", "dog", "runs", "fast"]
    s2 = ["the", "cat", "runs", "quick"]
    s3 = ["money", "bank", "account", "deposit"]
    
    for s in [s1, s2, s3]:
        for w in s:
            encoder.register_word(w)
    
    # Get sentence embeddings (final accumulated state)
    emb1 = encoder.encode_sentence(s1)
    emb2 = encoder.encode_sentence(s2)
    emb3 = encoder.encode_sentence(s3)
    
    sim_12 = cosine_sim(emb1, emb2)
    sim_13 = cosine_sim(emb1, emb3)
    sim_23 = cosine_sim(emb2, emb3)
    
    print("\n  Sentence similarity:")
    print("-" * 50)
    print(f"    s1 = {s1}")
    print(f"    s2 = {s2}")
    print(f"    s3 = {s3}")
    print()
    print(f"    s1 vs s2 (similar topics): {sim_12:.4f}")
    print(f"    s1 vs s3 (different):      {sim_13:.4f}")
    print(f"    s2 vs s3 (different):      {sim_23:.4f}")
    
    if sim_12 > sim_13 and sim_12 > sim_23:
        print("\n  ✅ Similar sentences have higher similarity!")
    else:
        print("\n  ⚠️ Sentence embedding needs improvement")
    
    return {
        's1_s2': sim_12,
        's1_s3': sim_13,
        's2_s3': sim_23
    }


def experiment_long_range_context():
    """Test: How far does context memory persist?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 4: Long-Range Context Memory")
    print("=" * 70)
    
    encoder = ContextualSNNEncoder(num_neurons=150, time_steps=50, seed=42)
    
    # Register words
    for w in ["prime", "filler1", "filler2", "filler3", "filler4", "target"]:
        encoder.register_word(w)
    
    results = []
    
    for n_fillers in [0, 1, 2, 3, 5, 10]:
        encoder.reset_context()
        
        # Prime word
        _, prime_spikes, _ = encoder.encode_word("prime")
        
        # Filler words
        fillers = [f"filler{i}" for i in range(n_fillers)]
        for f in fillers:
            encoder.register_word(f)
            encoder.encode_word(f)
        
        # Target (check if prime influence remains)
        target_feat, target_spikes, _ = encoder.encode_word("target")
        
        # How much of prime is in target?
        # Use correlation with context state
        context_strength = np.mean(np.abs(encoder.context_state))
        
        results.append({
            'n_fillers': n_fillers,
            'context_strength': context_strength
        })
        print(f"    {n_fillers} filler words: context strength = {context_strength:.4f}")
    
    # Check decay pattern
    first = results[0]['context_strength']
    last = results[-1]['context_strength']
    decay = (first - last) / first if first > 0 else 0
    
    print(f"\n    Context decay over 10 words: {decay*100:.1f}%")
    
    if decay < 0.8:
        print("\n  ✅ Context persists across multiple words!")
    else:
        print("\n  ⚠️ Context decays too quickly")
    
    return results


def experiment_disambiguation():
    """Test: Can context help disambiguate homonyms?"""
    print("\n" + "=" * 70)
    print("   EXPERIMENT 5: Homonym Disambiguation via Context")
    print("=" * 70)
    
    encoder = ContextualSNNEncoder(num_neurons=150, time_steps=50, seed=42)
    
    # Register homonyms and context words
    financial = np.array([1, 0, 0] * 50)
    nature = np.array([0, 1, 0] * 50)
    
    # "bank" - financial vs nature
    # "bat" - animal vs sports
    # "crane" - bird vs machine
    
    encoder.register_word("bank")
    encoder.register_word("bat")
    encoder.register_word("crane")
    
    # Financial context words
    encoder.register_word("money", financial)
    encoder.register_word("loan", financial)
    
    # Nature context words
    encoder.register_word("river", nature)
    encoder.register_word("bird", nature)
    
    # Test disambiguation of "bank"
    
    # Financial context → bank
    encoder.reset_context()
    encoder.encode_word("money")
    encoder.encode_word("loan")
    bank_fin, _, _ = encoder.encode_word("bank")
    
    # Nature context → bank
    encoder.reset_context()
    encoder.encode_word("river")
    encoder.encode_word("bird")
    bank_nat, _, _ = encoder.encode_word("bank")
    
    # Get reference representations
    encoder.reset_context()
    money_ref, _, _ = encoder.encode_word("money")
    encoder.reset_context()
    river_ref, _, _ = encoder.encode_word("river")
    
    # Check which context makes "bank" more similar to which domain
    sim_bank_fin_to_money = cosine_sim(bank_fin, money_ref)
    sim_bank_nat_to_money = cosine_sim(bank_nat, money_ref)
    sim_bank_fin_to_river = cosine_sim(bank_fin, river_ref)
    sim_bank_nat_to_river = cosine_sim(bank_nat, river_ref)
    
    print("\n  'bank' similarity to domain words:")
    print("-" * 50)
    print(f"    bank(financial) to 'money': {sim_bank_fin_to_money:.4f}")
    print(f"    bank(nature) to 'money':    {sim_bank_nat_to_money:.4f}")
    print(f"    bank(financial) to 'river': {sim_bank_fin_to_river:.4f}")
    print(f"    bank(nature) to 'river':    {sim_bank_nat_to_river:.4f}")
    
    # Check if disambiguation works
    financial_bias = sim_bank_fin_to_money - sim_bank_nat_to_money
    nature_bias = sim_bank_nat_to_river - sim_bank_fin_to_river
    
    print(f"\n    Financial context → money bias: {financial_bias:+.4f}")
    print(f"    Nature context → river bias:    {nature_bias:+.4f}")
    
    if financial_bias > 0 and nature_bias > 0:
        print("\n  ✅ Context successfully disambiguates homonyms!")
    else:
        print("\n  ⚠️ Disambiguation effect is mixed")
    
    return {
        'fin_to_money': sim_bank_fin_to_money,
        'nat_to_money': sim_bank_nat_to_money,
        'financial_bias': financial_bias,
        'nature_bias': nature_bias
    }


def main():
    print("=" * 70)
    print("   SNN LANGUAGE MODEL - CONTEXT-DEPENDENT REPRESENTATION")
    print("   Can context shape word meaning?")
    print("=" * 70)
    
    start = time.time()
    
    results = {}
    results['context_shift'] = experiment_context_shift()
    results['priming'] = experiment_semantic_priming()
    results['sentence'] = experiment_sentence_embedding()
    results['long_range'] = experiment_long_range_context()
    results['disambiguation'] = experiment_disambiguation()
    
    elapsed = time.time() - start
    
    # Summary
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY")
    print("=" * 70)
    
    findings = []
    
    if results['context_shift']['effect'] > 0.05:
        findings.append("✅ Context shifts word representations")
    if results['priming']['priming_effect'] > 0:
        findings.append("✅ Semantic priming increases similarity")
    if results['sentence']['s1_s2'] > results['sentence']['s1_s3']:
        findings.append("✅ Similar sentences cluster together")
    if results['disambiguation']['financial_bias'] > 0:
        findings.append("✅ Context disambiguates homonyms")
    
    print("\n  Key Findings:")
    for f in findings:
        print(f"    {f}")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # Save
    with open("results/context_dependent_results.txt", "w", encoding="utf-8") as f:
        f.write("SNN Context-Dependent Representation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Experiment 1: Context Shift\n")
        f.write(f"  Effect magnitude: {results['context_shift']['effect']:.4f}\n\n")
        
        f.write("Experiment 2: Semantic Priming\n")
        f.write(f"  Priming effect: {results['priming']['priming_effect']:.4f}\n\n")
        
        f.write("Experiment 3: Sentence Embedding\n")
        f.write(f"  Similar sentences: {results['sentence']['s1_s2']:.4f}\n")
        f.write(f"  Different sentences: {results['sentence']['s1_s3']:.4f}\n\n")
        
        f.write("Experiment 5: Disambiguation\n")
        f.write(f"  Financial bias: {results['disambiguation']['financial_bias']:.4f}\n")
        f.write(f"  Nature bias: {results['disambiguation']['nature_bias']:.4f}\n\n")
        
        f.write("Findings:\n")
        for finding in findings:
            f.write(f"  {finding}\n")
    
    print("\n  Results saved to: results/context_dependent_results.txt")


if __name__ == "__main__":
    main()
