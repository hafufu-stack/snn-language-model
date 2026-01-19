# SNN Language Model - Hybrid Spike + Membrane Potential Approach

🧠 **Spiking Neural Network for Character-Level Language Modeling**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 概要

SNNベースの文字レベル言語モデル。**スパイク数と膜電位の両方**を使用することで、従来のSNNアプローチより高精度を実現。

### 🔥 主な発見（2026年1月最新）

| 発見 | 結果 |
|------|------|
| 🔋 **エネルギー効率** | SNNはDNNの**14.7倍**効率的（スパース計算） |
| 📊 **最高精度** | SNN PPL=9.90 vs DNN=11.28 vs LSTM=15.67 |
| 🧬 **ハイブリッド効果** | 膜電位で**+39.7%**改善 |
| 💾 **スパース性** | わずか**7.6%**のニューロンが発火 |
| ⚡ **BitNet融合** | **Mixed Precisionで標準を超えた！** ✨NEW |

### 🚀 BitNet b1.58 + SNN（新発見！）

| Model | PPL | 備考 |
|-------|-----|------|
| **Mixed Precision (500n)** | **2.69** | **Standardを超えた！** ✅ |
| Standard SNN (200n) | 3.29 | baseline |

**キーポイント:**
- 入出力: 連続値（精度維持）
- リザーバ: 三値 {-1, 0, 1}（スパース性活用）
- 結果: 50-70%の演算が**加算のみ**で済む！

## 最新実験結果（v2）

### 完全比較：SNN vs DNN vs LSTM

| Model | Perplexity ↓ | Ops (M) | vs SNN Ops |
|-------|-------------|---------|------------|
| **SNN** | **9.90** | **478** | 1.0x |
| DNN | 11.28 | 674 | 1.41x |
| LSTM | 15.67 | 2683 | 5.61x |

→ **SNNが精度AND効率の両方で勝利！**

### ハイブリッドアブレーション

| モード | PPL | 改善率 |
|--------|-----|--------|
| Spike-only | 16.42 | 基準 |
| Membrane-only | 9.84 | +40.1% |
| **Hybrid** | **9.90** | **+39.7%** |

→ **膜電位が約40%の改善に貢献！**

### スパース計算効率

```
発火率: わずか 7.6% のニューロンが発火
Dense計算: 3213M ops → Sparse計算: 245M ops
削減率: 13.1倍！

エネルギー効率推定:
- SNN: 0.5 pJ/spike (ニューロモルフィックチップ)
- DNN: 5.0 pJ/op (CPU/GPU)
→ SNNは 14.7倍 エネルギー効率的！
```

## インストール

```bash
git clone https://github.com/hafufu-stack/snn-language-model.git
cd snn-language-model
pip install numpy
```

## 使い方

```bash
# コア実験
python experiments/core/snn_lm_prototype.py

# BitNet実験（Mixed Precision推奨）
python experiments/bitnet/snn_lm_bitnet_mixed_v3.py

# 高度な実験
python experiments/advanced/snn_lm_robustness.py
```

## ファイル構成

```
snn-language-model/
├── experiments/
│   ├── core/                  # コア実験
│   │   ├── snn_lm_prototype.py
│   │   ├── snn_lm_comparison.py
│   │   ├── snn_lm_benchmark.py
│   │   ├── snn_lm_sparse.py
│   │   └── snn_lm_hybrid_learning.py
│   ├── bitnet/                # BitNet融合実験 ✨NEW
│   │   ├── snn_lm_bitnet.py
│   │   ├── snn_lm_bitnet_mixed.py
│   │   ├── snn_lm_bitnet_mixed_v3.py  ← 最良
│   │   └── ...
│   └── advanced/              # 高度な実験
│       ├── snn_lm_robustness.py
│       ├── snn_lm_scaling.py
│       ├── snn_lm_innovative.py
│       └── ...
├── papers/                    # 論文
│   ├── paper_snn_lm.tex
│   └── paper_snn_lm_v2.tex
├── results/                   # 実験結果
└── README.md
```

## なぜSNNが優れているのか

### 1. エネルギー効率
- **スパース計算**: 発火したニューロンだけが計算に参加
- **イベント駆動**: 常時計算が不要

### 2. ノイズ耐性
- **閾値機構**: 小さなノイズはスパイクに変換されない
- **膜電位の平滑化**: 短期ノイズを吸収

### 3. 圧縮耐性 ✨NEW
- **80%ニューロン刈り込み**: それでもDNNより高品質
- **4bit量子化**: 8倍メモリ圧縮、+6.6%劣化のみ

### 4. BitNet融合 ✨NEW
- **三値重み**: {-1, 0, 1} で乗算不要
- **Mixed Precision**: 入出力連続+リザーバ三値が最適

## 論文

- **v2 (最新)**: [Zenodo DOI: 10.5281/zenodo.18294033](https://zenodo.org/records/18294033)
- v1: [Zenodo DOI: 10.5281/zenodo.18288582](https://doi.org/10.5281/zenodo.18288582)

## ライセンス

CC BY 4.0

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation) （日記や思いを発信）
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation) （プログラムの技術解説や構想を発信）
