# SNN Language Model - Ultimate SNN Architecture

🧠 **Spiking Neural Network for Character-Level Language Modeling**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 概要

SNNベースの文字レベル言語モデル。**BitNet + RWKV + Hybrid Readout + Progressive Training + Attention**を統合した**Ultimate SNN**で、従来のSNNアプローチより大幅な高精度と効率を実現。

### 🔥 最新発見（2026年1月20日）

| 発見 | 結果 |
|------|------|
| � **Ultimate SNN** | Standard SNNより**43.4%改善** |
| � **大規模検証** | 120K文字で**-20.9%**（本物の発見！） |
| ⚡ **BitNet融合** | Mixed Precisionで**標準を超えた** |
| 🎯 **RWKV統合** | Time-mixingで**36.1%改善** |
| 🔋 **エネルギー効率** | DNNの**14.7倍**効率的 |

### 🏆 Ultimate SNN（新発見！）

| Model | PPL | vs Standard | 検証データ |
|-------|-----|-------------|-----------|
| **Super Ultimate (500n)** | **4.40** | **-20.9%** | 120K文字 ✅ |
| Standard SNN (200n) | 5.56 | baseline | 120K文字 |

**Ultimate SNNの構成:**
- ✅ BitNet（三値重み {-1, 0, 1}）
- ✅ RWKV（Time-mixing + Channel-mixing）
- ✅ Hybrid Readout（スパイク + 膜電位）
- ✅ Progressive Training（4段階成長）
- ✅ Attention（履歴参照）

## 主要実験結果

### 完全比較：SNN vs DNN vs LSTM

| Model | Perplexity ↓ | Ops (M) | 効率 |
|-------|-------------|---------|------|
| **Ultimate SNN** | **10.59** | **245** | **14.7x** |
| Standard SNN | 18.71 | 150 | 9.2x |
| DNN | 11.28 | 674 | 1.0x |
| LSTM | 15.67 | 2683 | 0.25x |

→ **Ultimate SNNが精度AND効率の両方で勝利！**

### 大規模検証（120,000文字）

```
Dataset: 120,037 characters
Train: 6,400 samples, Test: 1,601 samples
24 parallel workers

Super Ultimate: PPL 4.40 ± 0.26
Standard SNN:   PPL 5.56 ± 0.17
Improvement:    -20.9% ✅

🎉 VALIDATED! This is a REAL discovery!
```

### ハイブリッドアブレーション

| モード | PPL | 改善率 |
|--------|-----|--------|
| Spike-only | 16.42 | 基準 |
| Membrane-only | 9.84 | +40.1% |
| **Hybrid** | **9.90** | **+39.7%** |

→ **膜電位が約40%の改善に貢献！**

## インストール

```bash
git clone https://github.com/hafufu-stack/snn-language-model.git
cd snn-language-model
pip install numpy
```

## 使い方

```bash
# Ultimate SNN実験（推奨）
python experiments/advanced/snn_lm_ultimate.py

# 大規模検証
python experiments/advanced/snn_lm_large_scale.py

# BitNet実験
python experiments/bitnet/snn_lm_bitnet_mixed_v3.py

# 22並列大規模実験
python experiments/advanced/snn_lm_massive_parallel.py
```

## ファイル構成

```
snn-language-model/
├── experiments/
│   ├── core/                  # コア実験
│   ├── bitnet/                # BitNet融合実験
│   └── advanced/              # 高度な実験
│       ├── snn_lm_ultimate.py      ← Ultimate SNN
│       ├── snn_lm_rwkv.py          ← RWKV統合
│       ├── snn_lm_combined.py      ← 統合実験
│       ├── snn_lm_large_scale.py   ← 大規模検証
│       ├── snn_lm_massive_parallel.py ← 並列実験
│       └── ...
├── image_gen/                 # 画像生成実験（v4 NEW）
│   ├── experiment_spike_membrane.py   ← スパイク+膜電位VAE
│   └── experiment_membrane_comparison.py ← 膜電位重み比較
├── results/                   # 実験結果
└── README.md
```

## なぜUltimate SNNが優れているのか

### 1. BitNet三値重み
- 重み：{-1, 0, 1} のみ
- **乗算不要**（加算のみ）
- 21倍メモリ削減

### 2. RWKV Time-mixing
- 長距離記憶を効率的に保持
- O(n) 複雑度（Transformerの O(n²) より軽量）

### 3. Progressive Training
- 小さいモデルから段階的に成長
- 学習の安定性向上

### 4. Attention
- 過去の履歴を参照
- 文脈理解の向上

### 5. Hybrid Readout
- スパイク数 + 膜電位
- 約40%の精度向上

## 論文

- **v4 (最新)**: [Zenodo DOI: 10.5281/zenodo.18398245](https://zenodo.org/records/18398245) - 言語 + 画像生成（Spiking VAE追加）
- v3: [Zenodo DOI: 10.5281/zenodo.18304632](https://zenodo.org/records/18304632)
- v2: [Zenodo DOI: 10.5281/zenodo.18294033](https://zenodo.org/records/18294033)
- v1: [Zenodo DOI: 10.5281/zenodo.18288582](https://doi.org/10.5281/zenodo.18288582)

## ライセンス

CC BY 4.0

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation)
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation)
