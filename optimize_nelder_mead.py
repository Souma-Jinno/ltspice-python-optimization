"""
Nelder-Mead法による等価回路パラメータ最適化

直接探索法（シンプレックス法）
- 特徴：勾配不要、初期値必要、初期値に依存
- scipy.optimize.minimize('Nelder-Mead')を使用
- SpiceLibを使用してLTSpiceシミュレーションを実行
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from utils import load_measurement_data, objective_function, save_results

# グローバル変数で評価回数と履歴を記録
evaluation_count = 0
error_history = []
best_error_history = []
best_error = float('inf')


def objective_with_history(params, freq, Z_meas):
    """
    履歴記録付きの目的関数（SpiceLibを使用）
    """
    global evaluation_count, error_history, best_error_history, best_error

    error = objective_function(params, freq, Z_meas, use_spicelib=True)

    evaluation_count += 1
    error_history.append(error)

    # 最良解の履歴も記録
    if error < best_error:
        best_error = error
    best_error_history.append(best_error)

    if evaluation_count % 10 == 0:
        print(f"Iteration {evaluation_count}: Error = {error:.6f}")

    return error


def main():
    global evaluation_count, error_history, best_error_history, best_error

    # グローバル変数をリセット
    evaluation_count = 0
    error_history = []
    best_error_history = []
    best_error = float('inf')

    # データの読み込み
    data_path = Path(__file__).parent / "data" / "capacitor_1port.csv"
    print(f"Loading data from: {data_path}")
    freq, Z_meas = load_measurement_data(data_path)
    print(f"Loaded {len(freq)} frequency points from {freq[0]/1e3:.1f} kHz to {freq[-1]/1e6:.1f} MHz\n")

    # パラメータの初期値
    # 初期値: R=0.1Ω, L=10nH, C=100nF
    x0 = np.array([0.1, 10e-9, 100e-9])

    print("="*60)
    print("Nelder-Mead Optimization (SpiceLib)")
    print("="*60)
    print(f"Initial parameters:")
    print(f"  R = {x0[0]:.6f} Ω")
    print(f"  L = {x0[1]:.6e} H")
    print(f"  C = {x0[2]:.6e} F")
    print("\nStarting optimization...")
    print("-"*60)

    # Nelder-Mead法で最適化
    result = minimize(
        objective_with_history,
        x0,
        args=(freq, Z_meas),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9}
    )

    print("-"*60)
    print(f"\nOptimization completed!")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"\nOptimized parameters:")
    print(f"  R = {result.x[0]:.6f} Ω")
    print(f"  L = {result.x[1]:.6e} H")
    print(f"  C = {result.x[2]:.6e} F")
    print(f"\nFinal error: {result.fun:.6f}")

    # 結果を保存（同じフォルダ内のresults）
    output_dir = Path(__file__).parent / "results"
    save_results(
        output_dir=output_dir,
        method_name="NelderMead",
        result={'x': result.x, 'fun': result.fun},
        freq=freq,
        Z_meas=Z_meas,
        use_spicelib=True,
        history=error_history,
        best_history=best_error_history
    )


if __name__ == "__main__":
    main()
