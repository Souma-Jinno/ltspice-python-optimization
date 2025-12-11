# 等価回路パラメータ最適化 - シミュレーションコード

月刊EMC誌「Python-LTSpice統合による等価回路最適化の実践」で使用したシミュレーションコード

## 概要

本コードはSpiceLibを使用してLTSpiceと連携し、等価回路パラメータ（R, L, C）の最適化を行います。
VNA測定データとLTSpiceシミュレーション結果を比較し、誤差を最小化するパラメータを探索します。

## フォルダ構成

```
simulation/
├── data/                    # 測定データ・回路データ
│   ├── capacitor_1port.csv  # VNA測定データ（S11パラメータ）
│   └── RLC.net              # LTSpiceネットリスト
├── results/                 # 最適化結果の出力先
├── optimize_de.py           # 差分進化法（推奨）
├── optimize_lbfgsb.py       # L-BFGS-B（勾配ベース）
├── optimize_nelder_mead.py  # Nelder-Mead法
├── optimize_nsga2.py        # NSGA-II（多目的最適化）
├── optimize_pso.py          # 粒子群最適化
├── optimize_cmaes.py        # CMA-ES
├── optimize_bayesian.py     # ベイズ最適化
└── utils.py                 # 共通ユーティリティ（SpiceLib連携）
```

resultsフォルダには以下のファイルが生成されます：
- `*_convergence.png` - 目的関数の収束履歴（全評価と最良解）
- `*_impedance_comparison.png` - 測定値とシミュレーションの比較
- `*_parameters.csv` - 最適化されたパラメータ
- `*_results.txt` - 詳細な結果サマリー

## 必要環境

### LTSpice

- LTSpice XVII以上がインストールされていること
- macOSの場合: `/Applications/LTspice.app` にインストール

### Pythonパッケージ

```bash
pip install numpy scipy matplotlib pandas
pip install spicelib  # LTSpice連携用
pip install pymoo pyswarms cma scikit-optimize
```

## 使用方法

### 差分進化法で最適化（推奨）

```bash
cd simulation
python optimize_de.py
```

### 他の手法で最適化

```bash
cd simulation
python optimize_lbfgsb.py       # L-BFGS-B
python optimize_nelder_mead.py  # Nelder-Mead
python optimize_nsga2.py        # NSGA-II
python optimize_pso.py          # PSO
python optimize_cmaes.py        # CMA-ES
python optimize_bayesian.py     # ベイズ最適化
```

## SpiceLibによるLTSpice連携

本コードはSpiceLibを使用してLTSpiceシミュレーションを自動実行します：

1. `data/RLC.net` のネットリストを読み込み
2. パラメータ（R, L, C）を動的に変更
3. LTSpiceでACシミュレーションを実行
4. 結果から複素インピーダンスを計算

```python
# utils.py での実装例
from spicelib import SpiceEditor, SimRunner, RawRead

editor = SpiceEditor(str(netlist_path))
editor.set_parameter('R', f"{R}")
editor.set_parameter('L', f"{L}")
editor.set_parameter('C', f"{C}")
# ...
```

## 測定データ

`data/capacitor_1port.csv` はVNAで測定したコンデンサのS11パラメータです：
- 周波数範囲: 9 kHz ～ 500 MHz
- 測定点数: 2001点
- 形式: 周波数, Re(S11), Im(S11)

## LTSpiceネットリスト

`data/RLC.net` は直列RLC等価回路のネットリストです：

```spice
* RLC Circuit for Parameter Extraction
V1 N001 0 AC 1
R1 N002 N001 {R}
L1 N002 N003 {L}
C1 N003 0 {C}
.param R=1 L=1e-8 C=1e-9
.ac dec 100 9k 500Meg
```

## 最適化手法の比較

| 手法 | 特徴 | 初期値 |
|------|------|--------|
| 差分進化法 | 大域探索、ノイズ耐性（**推奨**） | 不要 |
| L-BFGS-B | 高速収束、局所解に陥りやすい | 必要 |
| Nelder-Mead | 勾配不要、初期値に依存 | 必要 |
| CMA-ES | 高速収束、連続最適化 | 必要 |
| PSO | 大域探索、高速収束 | 不要 |
| NSGA-II | 多目的最適化対応 | 不要 |
| ベイズ最適化 | 評価回数最小化 | 不要 |

## 開発支援

本プロジェクトでは [Claude Code](https://claude.ai/claude-code) を開発支援に使用しています。

## ライセンス

MIT License

## 参考文献

神野崇馬, "Python-LTSpice統合による等価回路最適化の実践", 月刊EMC, 2025.
