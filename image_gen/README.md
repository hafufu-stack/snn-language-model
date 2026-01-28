# Image Generation with Spiking Neural Networks

This folder contains experiments for image generation using SNNs with the spike+membrane potential hybrid approach.

## Key Findings (v4)

- **50% membrane weight** is optimal for Spiking VAE
- Achieves **96% sparsity** with **57% loss reduction** compared to spike-only
- Solves **posterior collapse** problem (KL=0) in Spiking VAEs

## Files

- `experiment_spike_membrane.py`: Main experiment with 30% membrane weight
- `experiment_membrane_comparison.py`: Comparison of 50% vs 70% membrane weights

## Usage

```bash
python experiment_membrane_comparison.py
```

Results will be saved in `output_membrane_comparison/`.
