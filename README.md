# SNN Language Model - Hybrid Spike + Membrane Potential Approach

🧠 **Spiking Neural Network for Character-Level Language Modeling**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 概要

SNNベースの文字レベル言語モデル。**スパイク数と膜電位の両方**を使用することで、従来のSNNアプローチより高精度を実現。

### 主な発見

| 発見 | 結果 |
|------|------|
| 🔋 **エネルギー効率** | SNNはDNNの**42倍**効率的 |
| 🛡️ **ノイズ耐性** | 30%ノイズでも精度低下なし |
| 🧬 **生物学的妥当性** | 脳の意思決定メカニズムに近い |

## 背景

従来のSNNはスパイク数（発火回数）だけで出力を決定していました。しかし、**膜電位**（ニューロンの内部状態）にも重要な情報が含まれています。

```python
# 従来: スパイク数だけ
output = W @ spike_count

# 本研究: スパイク + 膜電位
output = W_spike @ spike_count + W_membrane @ membrane_potential
```

## 実験結果

### 精度比較（22,920文字で学習）

| モデル | 精度 | Perplexity | 演算数 |
|--------|------|-----------|--------|
| SNN-300 | 10.24% | 28.20 | 93.85M |
| DNN-300 | 15.22% | 46.27 | 5,885M |
| LSTM-300 | 18.77% | 41.28 | 23,542M |

### ノイズ耐性

| モデル | 0%ノイズ | 30%ノイズ | 劣化 |
|--------|---------|----------|------|
| **SNN-500** | 8.67% | 8.80% | **-0.13%** ✅ |
| DNN-500 | 10.83% | 7.77% | +3.06% ⚠️ |
| LSTM-300 | 18.77% | 17.71% | +1.05% |

→ **SNNはノイズがあっても精度が落ちない！**

### エネルギー効率

| モデル | 効率 (精度/百万演算) |
|--------|----------------------|
| **SNN-300** | **0.1091** |
| DNN-300 | 0.0026 |
| LSTM-300 | 0.0008 |

→ **SNNはDNNの42倍、LSTMの136倍効率的！**

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
├── snn_lm_prototype.py      # プロトタイプ（Hybrid vs Traditional）
├── snn_lm_comparison.py     # アーキテクチャ比較
├── snn_lm_large_scale.py    # 大規模実験
├── snn_lm_parallel.py       # 並列処理版
└── results/                 # 実験結果
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

Coming soon on Zenodo / arXiv

## ライセンス

CC BY 4.0

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation) （日記や思いを発信）
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation) （プログラムの技術解説や構想を発信）

