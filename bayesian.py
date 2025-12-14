"""
ベイズ最適化による等価回路パラメータ最適化

ガウス過程ベース
- 特徴：評価回数最小化、高コスト問題に有効
- scikit-optimize (gp_minimize)を使用
- SpiceLibを使用してLTSpiceシミュレーションを実行
"""
import numpy as np
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real

from utils import load_measurement_data, objective_function, save_results

# グローバル変数で履歴を記録
error_history = []
best_error_history = []
best_error = float('inf')
freq_global = None
Z_meas_global = None


def objective_bayesian(params):
    """
    ベイズ最適化用の目的関数（SpiceLibを使用）

    Parameters:
    -----------
    params : list [R, L, C]
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

    if len(error_history) % 10 == 0:
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

    # パラメータの探索範囲
    # R (0.001~10Ω), L (1nH~1μH), C (1pF~10μF)
    space = [
        Real(0.001, 10.0, name='R', prior='log-uniform'),      # R [Ω]
        Real(1e-9, 1e-6, name='L', prior='log-uniform'),       # L [H]
        Real(1e-12, 10e-6, name='C', prior='log-uniform')      # C [F]
    ]

    print("="*60)
    print("Bayesian Optimization (SpiceLib)")
    print("="*60)
    print(f"Search space:")
    print(f"  R: [0.001, 10.0] Ω (log-uniform)")
    print(f"  L: [1e-9, 1e-6] H (log-uniform)")
    print(f"  C: [1e-12, 1e-5] F (log-uniform)")
    print("\nAlgorithm parameters:")
    print(f"  Number of calls: 100")
    print(f"  Number of initial points: 10")
    print("\nStarting optimization...")
    print("-"*60)

    # ベイズ最適化を実行
    result = gp_minimize(
        objective_bayesian,
        space,
        n_calls=100,
        n_initial_points=10,
        acq_func='EI',  # Expected Improvement
        random_state=42,
        verbose=False
    )

    print("-"*60)
    print(f"\nOptimization completed!")
    print(f"Number of function evaluations: {len(result.func_vals)}")
    print(f"\nOptimized parameters:")
    print(f"  R = {result.x[0]:.6f} Ω")
    print(f"  L = {result.x[1]:.6e} H")
    print(f"  C = {result.x[2]:.6e} F")
    print(f"\nFinal error: {result.fun:.6f}")

    # 結果を保存（同じフォルダ内のresults）
    output_dir = Path(__file__).parent / "results"
    save_results(
        output_dir=output_dir,
        method_name="Bayesian",
        result={'x': result.x, 'fun': result.fun},
        freq=freq,
        Z_meas=Z_meas,
        use_spicelib=True,
        history=error_history,
        best_history=best_error_history
    )


if __name__ == "__main__":
    main()
