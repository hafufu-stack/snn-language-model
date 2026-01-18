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
| ⚡ **効率スケーリング** | 長いシーケンスで効率向上（1.48x→1.52x）|

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

### スケーリング特性

| Seq Length | Efficiency |
|------------|------------|
| 10 | 1.48x |
| 20 | 1.50x |
| 40 | 1.52x |
| 80 | 1.52x |

→ **長いシーケンスで効率向上！**



## インストール

```bash
git clone https://github.com/hafufu-stack/snn-language-model.git
cd snn-language-model
pip install numpy
```

## 使い方

```bash
# プロトタイプ実験（Traditional vs Hybrid）
python snn_lm_prototype.py

# アーキテクチャ比較（SNN vs DNN vs LSTM）
python snn_lm_comparison.py

# 大規模実験（300-600ニューロン）
python snn_lm_large_scale.py

# 並列処理版（高速）
python snn_lm_parallel.py
```

## ファイル構成

```
snn-language-model/
├── snn_lm_prototype.py       # プロトタイプ（Hybrid vs Traditional）
├── snn_lm_comparison.py      # アーキテクチャ比較
├── snn_lm_large_scale.py     # 大規模実験
├── snn_lm_parallel.py        # 並列処理版
├── snn_lm_sparse.py          # スパース計算ベンチマーク ✨NEW
├── snn_lm_hybrid_learning.py # ハイブリッド効果検証 ✨NEW
├── snn_lm_scaling.py         # スケーリング特性 ✨NEW
├── snn_lm_temporal_v2.py     # 時間符号化実験 ✨NEW
├── snn_lm_overcomplete.py    # オーバーコンプリート表現 ✨NEW
├── snn_lm_context.py         # 文脈依存表現 ✨NEW
├── snn_lm_advanced_v2.py     # 論文v2用実験 ✨NEW
└── results/                  # 実験結果
```

## なぜSNNが優れているのか

### 1. エネルギー効率
- **スパース計算**: 発火したニューロンだけが計算に参加
- **イベント駆動**: 常時計算が不要

### 2. ノイズ耐性
- **閾値機構**: 小さなノイズはスパイクに変換されない
- **膜電位の平滑化**: 短期ノイズを吸収

### 3. 生物学的妥当性
- **デジタル（スパイク）+ アナログ（膜電位）** の両方を使用
- 脳の意思決定プロセスに近い

## 論文

Funasaki, H. (2026). Hybrid Spiking Language Model: Combining Spike Counts and Membrane Potentials for Energy-Efficient and Noise-Robust Character Prediction. Zenodo. https://doi.org/10.5281/zenodo.18288582

## ライセンス

CC BY 4.0

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation) （日記や思いを発信）
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation) （プログラムの技術解説や構想を発信）

