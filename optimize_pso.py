"""
PSO（粒子群最適化）による等価回路パラメータ最適化

進化的アルゴリズム
- 特徴：大域探索、初期値不要、高速収束
- PySwarms (GlobalBestPSO)を使用
- SpiceLibを使用してLTSpiceシミュレーションを実行
"""
import numpy as np
from pathlib import Path
from pyswarms.single.global_best import GlobalBestPSO

from utils import load_measurement_data, objective_function, save_results

# グローバル変数で履歴を記録
error_history = []
best_error_history = []
best_error = float('inf')
freq_global = None
Z_meas_global = None


def objective_pso(X):
    """
    PySwarms用の目的関数（複数の粒子を同時に評価、SpiceLibを使用）

    Parameters:
    -----------
    X : ndarray, shape (n_particles, n_dimensions)
        粒子の位置

    Returns:
    --------
    errors : ndarray, shape (n_particles,)
        各粒子の誤差
    """
    global freq_global, Z_meas_global, best_error

    errors = np.array([objective_function(x, freq_global, Z_meas_global, use_spicelib=True) for x in X])

    # 全評価の履歴を記録
    for e in errors:
        error_history.append(e)
        # 最良解の履歴も記録
        if e < best_error:
            best_error = e
        best_error_history.append(best_error)

    if len(best_error_history) % 100 == 0:
        print(f"Evaluation {len(best_error_history)}: Best Error = {best_error:.6f}")

    return errors


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

    # パラメータの境界
    # R (0.001~10Ω), L (1nH~1μH), C (1pF~10μF)
    bounds = (
        np.array([0.001, 1e-9, 1e-12]),  # 下限
        np.array([10.0, 1e-6, 10e-6])    # 上限
    )

    print("="*60)
    print("PSO (Particle Swarm Optimization) (SpiceLib)")
    print("="*60)
    print(f"Bounds:")
    print(f"  R: [{bounds[0][0]:.3f}, {bounds[1][0]:.1f}] Ω")
    print(f"  L: [{bounds[0][1]:.1e}, {bounds[1][1]:.1e}] H")
    print(f"  C: [{bounds[0][2]:.1e}, {bounds[1][2]:.1e}] F")
    print("\nAlgorithm parameters:")
    print(f"  Number of particles: 30")
    print(f"  Max iterations: 100")
    print("\nStarting optimization...")
    print("-"*60)

    # PSOオプション
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # PSOオプティマイザーを初期化
    optimizer = GlobalBestPSO(
        n_particles=30,
        dimensions=3,
        options=options,
        bounds=bounds
    )

    # 最適化を実行
    cost, pos = optimizer.optimize(objective_pso, iters=100)

    print("-"*60)
    print(f"\nOptimization completed!")
    print(f"Number of evaluations: {len(error_history)}")
    print(f"\nOptimized parameters:")
    print(f"  R = {pos[0]:.6f} Ω")
    print(f"  L = {pos[1]:.6e} H")
    print(f"  C = {pos[2]:.6e} F")
    print(f"\nFinal error: {cost:.6f}")

    # 結果を保存（同じフォルダ内のresults）
    output_dir = Path(__file__).parent / "results"
    save_results(
        output_dir=output_dir,
        method_name="PSO",
        result={'x': pos, 'fun': cost},
        freq=freq,
        Z_meas=Z_meas,
        use_spicelib=True,
        history=error_history,
        best_history=best_error_history
    )


if __name__ == "__main__":
    main()
