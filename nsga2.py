"""
NSGA-II（遺伝的アルゴリズム）による等価回路パラメータ最適化

進化的アルゴリズム
- 特徴：多目的最適化対応、大域探索、集団ベース
- pymoo (NSGA-II)を使用
- SpiceLibを使用してLTSpiceシミュレーションを実行
"""
import numpy as np
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from utils import load_measurement_data, objective_function, save_results

# グローバル変数で履歴を記録
error_history = []
best_error_history = []
best_error = float('inf')


class RLCOptimizationProblem(Problem):
    """
    pymoo用の最適化問題クラス
    """
    def __init__(self, freq, Z_meas):
        # パラメータの境界
        # R (0.001~10Ω), L (1nH~1μH), C (1pF~10μF)
        xl = np.array([0.001, 1e-9, 1e-12])
        xu = np.array([10.0, 1e-6, 10e-6])

        super().__init__(
            n_var=3,                # 変数の数（R, L, C）
            n_obj=1,                # 目的関数の数
            xl=xl,                  # 下限
            xu=xu                   # 上限
        )

        self.freq = freq
        self.Z_meas = Z_meas
        self.eval_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """
        各個体に対して目的関数を評価（SpiceLibを使用）
        """
        global best_error
        F = []
        for x in X:
            error = objective_function(x, self.freq, self.Z_meas, use_spicelib=True)
            F.append(error)
            error_history.append(error)
            self.eval_count += 1

            # 最良解の履歴も記録
            if error < best_error:
                best_error = error
            best_error_history.append(best_error)

        if self.eval_count % 50 == 0:
            print(f"Evaluation {self.eval_count}: Best Error = {min(error_history):.6f}")

        out["F"] = np.array(F).reshape(-1, 1)


def main():
    global error_history, best_error_history, best_error

    # グローバル変数をリセット
    error_history = []
    best_error_history = []
    best_error = float('inf')

    # データの読み込み
    data_path = Path(__file__).parent / "data" / "capacitor_1port.csv"
    print(f"Loading data from: {data_path}")
    freq, Z_meas = load_measurement_data(data_path)
    print(f"Loaded {len(freq)} frequency points from {freq[0]/1e3:.1f} kHz to {freq[-1]/1e6:.1f} MHz\n")

    print("="*60)
    print("NSGA-II Optimization (SpiceLib)")
    print("="*60)
    print(f"Bounds:")
    print(f"  R: [0.001, 10.0] Ω")
    print(f"  L: [1e-9, 1e-6] H")
    print(f"  C: [1e-12, 1e-5] F")
    print("\nAlgorithm parameters:")
    print(f"  Population size: 20")
    print(f"  Max generations: 30")
    print("\nStarting optimization...")
    print("-"*60)

    # 最適化問題を定義
    problem = RLCOptimizationProblem(freq, Z_meas)

    # NSGA-IIアルゴリズムを設定
    algorithm = NSGA2(pop_size=20)

    # 最適化を実行
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 30),
        seed=42,
        verbose=False
    )

    print("-"*60)
    print(f"\nOptimization completed!")
    print(f"Number of function evaluations: {problem.eval_count}")
    print(f"\nOptimized parameters:")
    print(f"  R = {res.X[0]:.6f} Ω")
    print(f"  L = {res.X[1]:.6e} H")
    print(f"  C = {res.X[2]:.6e} F")
    print(f"\nFinal error: {res.F[0]:.6f}")

    # 結果を保存（同じフォルダ内のresults）
    output_dir = Path(__file__).parent / "results"
    save_results(
        output_dir=output_dir,
        method_name="NSGA2",
        result={'x': res.X, 'fun': res.F[0]},
        freq=freq,
        Z_meas=Z_meas,
        use_spicelib=True,
        history=error_history,
        best_history=best_error_history
    )


if __name__ == "__main__":
    main()
