"""
CMA-ES（共分散行列適応進化戦略）による等価回路パラメータ最適化

進化的アルゴリズム
- 特徴：共分散行列適応、高速収束、連続最適化
- cma (pycma)を使用
- SpiceLibを使用してLTSpiceシミュレーションを実行
"""
import numpy as np
from pathlib import Path
import cma

from utils import load_measurement_data, objective_function, save_results

# グローバル変数で履歴を記録
error_history = []
best_error_history = []
best_error = float('inf')
freq_global = None
Z_meas_global = None


def objective_cmaes(params):
    """
    CMA-ES用の目的関数（SpiceLibを使用）

    Parameters:
    -----------
    params : array-like [R, L, C]
        最適化パラメータ

    Returns:
    --------
    error : float
        誤差
    """
    global freq_global, Z_meas_global, best_error

    error = objective_function(params, freq_global, Z_meas_global, use_spicelib=True)
    error_history.append(error)

    # 最良解の履歴も記録
    if error < best_error:
        best_error = error
    best_error_history.append(best_error)

    if len(error_history) % 50 == 0:
        print(f"Evaluation {len(error_history)}: Error = {error:.6f}")

    return error


def main():
    global freq_global, Z_meas_global, error_history, best_error_history, best_error

    # グローバル変数をリセット
    error_history = []
    best_error_history = []
    best_error = float('inf')

    # データの読み込み
    data_path = Path(__file__).parent / "data" / "capacitor_1port.csv"
    print(f"Loading data from: {data_path}")
    freq, Z_meas = load_measurement_data(data_path)
    print(f"Loaded {len(freq)} frequency points from {freq[0]/1e3:.1f} kHz to {freq[-1]/1e6:.1f} MHz\n")

    # グローバル変数に設定
    freq_global = freq
    Z_meas_global = Z_meas

    # 初期値（対数空間の中央値）
    # R: 0.1Ω, L: 30nH, C: 100nF
    x0 = [0.1, 30e-9, 100e-9]

    # 初期標準偏差（探索範囲の1/6程度）
    sigma0 = 0.3

    # パラメータの境界
    # R (0.001~10Ω), L (1nH~1μH), C (1pF~10μF)
    bounds = [
        [0.001, 10.0],      # R [Ω]
        [1e-9, 1e-6],       # L [H]
        [1e-12, 10e-6]      # C [F]
    ]

    print("="*60)
    print("CMA-ES Optimization (SpiceLib)")
    print("="*60)
    print(f"Initial parameters:")
    print(f"  R = {x0[0]:.6f} Ω")
    print(f"  L = {x0[1]:.6e} H")
    print(f"  C = {x0[2]:.6e} F")
    print(f"\nBounds:")
    print(f"  R: [{bounds[0][0]:.3f}, {bounds[0][1]:.1f}] Ω")
    print(f"  L: [{bounds[1][0]:.1e}, {bounds[1][1]:.1e}] H")
    print(f"  C: [{bounds[2][0]:.1e}, {bounds[2][1]:.1e}] F")
    print(f"\nAlgorithm parameters:")
    print(f"  Initial sigma: {sigma0}")
    print("\nStarting optimization...")
    print("-"*60)

    # CMA-ESオプション
    opts = {
        'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
        'tolfun': 1e-9,
        'maxiter': 100,
        'verb_disp': 0,
        'verbose': -1,
        'seed': 42
    }

    # CMA-ES最適化を実行
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(objective_cmaes)

    # 最適解を取得
    result = es.result
    x_opt = result.xbest
    f_opt = result.fbest

    print("-"*60)
    print(f"\nOptimization completed!")
    print(f"Number of function evaluations: {result.evaluations}")
    print(f"\nOptimized parameters:")
    print(f"  R = {x_opt[0]:.6f} Ω")
    print(f"  L = {x_opt[1]:.6e} H")
    print(f"  C = {x_opt[2]:.6e} F")
    print(f"\nFinal error: {f_opt:.6f}")

    # 結果を保存（同じフォルダ内のresults）
    output_dir = Path(__file__).parent / "results"
    save_results(
        output_dir=output_dir,
        method_name="CMAES",
        result={'x': x_opt, 'fun': f_opt},
        freq=freq,
        Z_meas=Z_meas,
        use_spicelib=True,
        history=error_history,
        best_history=best_error_history
    )


if __name__ == "__main__":
    main()
