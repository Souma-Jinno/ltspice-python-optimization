"""
差分進化法による等価回路パラメータ最適化（実数部・虚数部目的関数）

進化的アルゴリズム
- 特徴：大域探索、初期値不要、ノイズ耐性
- scipy.optimize.differential_evolutionを使用
- 目的関数：実数部と虚数部の対数スケール誤差
"""
import numpy as np
import sys
from pathlib import Path
from scipy.optimize import differential_evolution

# 親ディレクトリのutilsをインポート
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_measurement_data, objective_function_real_imag, save_results

# グローバル変数で評価回数と履歴を記録
evaluation_count = 0
error_history = []


def objective_with_history(params, freq, Z_meas):
    """
    履歴記録付きの目的関数（実数部・虚数部）
    """
    global evaluation_count, error_history

    error = objective_function_real_imag(params, freq, Z_meas)

    evaluation_count += 1
    error_history.append(error)

    if evaluation_count % 50 == 0:
        print(f"Evaluation {evaluation_count}: Error = {error:.6f}")

    return error


def main():
    # データの読み込み
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "capacitor_1port.csv"
    print(f"Loading data from: {data_path}")
    freq, Z_meas = load_measurement_data(data_path)
    print(f"Loaded {len(freq)} frequency points from {freq[0]/1e3:.1f} kHz to {freq[-1]/1e6:.1f} MHz\n")

    # パラメータの境界（初期値不要）
    # R (0.001~10Ω), L (1nH~1μH), C (1pF~10μF)
    bounds = [
        (0.001, 10.0),      # R [Ω]
        (1e-9, 1e-6),       # L [H]
        (1e-12, 10e-6)      # C [F]
    ]

    print("="*60)
    print("Differential Evolution Optimization (Real-Imag)")
    print("="*60)
    print("Objective function: Real and Imaginary parts (log scale)")
    print(f"Bounds:")
    print(f"  R: [{bounds[0][0]:.3f}, {bounds[0][1]:.1f}] Ω")
    print(f"  L: [{bounds[1][0]:.1e}, {bounds[1][1]:.1e}] H")
    print(f"  C: [{bounds[2][0]:.1e}, {bounds[2][1]:.1e}] F")
    print("\nAlgorithm parameters:")
    print(f"  Population size: 15 (5 × 3 parameters)")
    print(f"  Max iterations: 50")
    print("\nStarting optimization...")
    print("-"*60)

    # 差分進化法で最適化
    result = differential_evolution(
        objective_with_history,
        bounds,
        args=(freq, Z_meas),
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        tol=1e-9,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False
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

    # 結果を保存（実数部・虚数部プロットを使用）
    output_dir = Path(__file__).parent / "results_real_imag"
    save_results(
        output_dir=output_dir,
        method_name="DifferentialEvolution_RealImag",
        result={'x': result.x, 'fun': result.fun},
        freq=freq,
        Z_meas=Z_meas,
        history=error_history,
        plot_type='real_imag'
    )


if __name__ == "__main__":
    main()
