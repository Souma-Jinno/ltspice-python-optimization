"""
共通のユーティリティ関数

SpiceLibを使用したLTSpiceシミュレーション、目的関数、プロット関数など
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import glob

# SpiceLibのインポート
try:
    from spicelib import SpiceEditor, RawRead
    from PyLTSpice import SimRunner
    SPICELIB_AVAILABLE = True
except ImportError:
    print("Warning: SpiceLib is not installed. Install with: pip install spicelib PyLTSpice")
    SPICELIB_AVAILABLE = False


def s11_to_impedance(s11_db, s11_deg, z0=50.0):
    """
    S11パラメータ（dB, degree）からインピーダンスを計算

    Parameters:
    -----------
    s11_db : float or array-like
        S11の振幅 [dB]
    s11_deg : float or array-like
        S11の位相 [degree]
    z0 : float
        特性インピーダンス（デフォルト: 50Ω）

    Returns:
    --------
    Z : complex or array-like
        インピーダンス [Ω]
    """
    # dBを線形スケールに変換
    s11_mag = 10 ** (s11_db / 20)

    # 度をラジアンに変換
    s11_rad = np.deg2rad(s11_deg)

    # 複素S11を計算
    s11_complex = s11_mag * np.exp(1j * s11_rad)

    # インピーダンスを計算: Z = Z0 * (1 + S11) / (1 - S11)
    Z = z0 * (1 + s11_complex) / (1 - s11_complex)

    return Z


def load_measurement_data(csv_path):
    """
    測定データ（CSV）を読み込み

    Parameters:
    -----------
    csv_path : str or Path
        CSVファイルのパス

    Returns:
    --------
    freq : ndarray
        周波数 [Hz]
    Z_meas : ndarray (complex)
        測定インピーダンス [Ω]
    """
    # CSVの読み込み（ヘッダーをスキップ）
    df = pd.read_csv(csv_path, skiprows=6)

    # 周波数とS11パラメータを抽出（型を明示的に変換）
    freq = pd.to_numeric(df['Freq(Hz)'], errors='coerce').values
    s11_db = pd.to_numeric(df['S11(DB)'], errors='coerce').values
    s11_deg = pd.to_numeric(df['S11(DEG)'], errors='coerce').values

    # NaNを除去
    valid_idx = ~(np.isnan(freq) | np.isnan(s11_db) | np.isnan(s11_deg))
    freq = freq[valid_idx]
    s11_db = s11_db[valid_idx]
    s11_deg = s11_deg[valid_idx]

    # インピーダンスに変換
    Z_meas = s11_to_impedance(s11_db, s11_deg)

    return freq, Z_meas


def simulate_rlc_impedance_spicelib(freq, R, L, C, netlist_path=None):
    """
    SpiceLibを使用してRLC回路のインピーダンスをシミュレーション

    Parameters:
    -----------
    freq : ndarray
        周波数 [Hz]
    R : float
        抵抗 [Ω]
    L : float
        インダクタンス [H]
    C : float
        キャパシタンス [F]
    netlist_path : str or Path, optional
        ネットリストファイルのパス（デフォルト: data/RLC.net）

    Returns:
    --------
    Z : ndarray (complex)
        インピーダンス [Ω]
    """
    if not SPICELIB_AVAILABLE:
        raise ImportError("SpiceLib is not installed")

    # ネットリストファイルのデフォルトパス
    if netlist_path is None:
        netlist_path = Path(__file__).parent / "data" / "RLC.net"

    netlist_path = Path(netlist_path)

    if not netlist_path.exists():
        raise FileNotFoundError(f"Netlist file not found: {netlist_path}")

    # 一時ディレクトリを作成（シミュレーション結果を保存）
    with tempfile.TemporaryDirectory() as tmpdir:
        # ネットリストをコピーして編集
        temp_netlist = Path(tmpdir) / "RLC_temp.net"

        # SpiceEditorでネットリストを読み込み
        editor = SpiceEditor(str(netlist_path))

        # パラメータを設定
        editor.set_parameter('R', f"{R}")
        editor.set_parameter('L', f"{L}")
        editor.set_parameter('C', f"{C}")

        # 一時ネットリストに保存
        editor.save_netlist(str(temp_netlist))

        # シミュレーションを実行（SimRunnerを使用）
        runner = SimRunner(output_folder=tmpdir, verbose=False)
        runner.run(str(temp_netlist))
        runner.wait_completion()

        # 結果を読み込み（.rawファイルを検索）
        raw_files = glob.glob(os.path.join(tmpdir, '*.raw'))
        if not raw_files:
            raise RuntimeError(f"Simulation failed. No .raw file found in {tmpdir}")

        raw_data = RawRead(raw_files[0])

        # 周波数を取得（実数部のみ）
        freq_sim = np.real(raw_data.get_trace('frequency').get_wave())

        # 電圧と電流を取得
        v1 = raw_data.get_trace('V(N001)').get_wave()  # 入力電圧
        i1 = raw_data.get_trace('I(V1)').get_wave()     # 電源電流

        # インピーダンスを計算: Z = V / I
        # 注: LTSpiceではI(V1)は電圧源から流れ出る電流として定義されている
        #     回路に流れ込む電流を基準とするため、符号を反転する
        Z_sim = -v1 / i1

        # 周波数の補間（測定周波数に合わせる）
        Z_interpolated = np.interp(freq, freq_sim, Z_sim.real) + \
                        1j * np.interp(freq, freq_sim, Z_sim.imag)

    return Z_interpolated


def simulate_rlc_impedance_analytical(freq, R, L, C):
    """
    解析的にRLC回路のインピーダンスを計算（SpiceLibのバックアップ）

    Parameters:
    -----------
    freq : ndarray
        周波数 [Hz]
    R : float
        抵抗 [Ω]
    L : float
        インダクタンス [H]
    C : float
        キャパシタンス [F]

    Returns:
    --------
    Z : ndarray (complex)
        インピーダンス [Ω]
    """
    omega = 2 * np.pi * freq

    # Z = R + j*omega*L + 1/(j*omega*C)
    Z = R + 1j * omega * L + 1 / (1j * omega * C)

    return Z


def simulate_rlc_impedance(freq, R, L, C, use_spicelib=True, netlist_path=None):
    """
    RLC回路のインピーダンスをシミュレーション

    Parameters:
    -----------
    freq : ndarray
        周波数 [Hz]
    R : float
        抵抗 [Ω]
    L : float
        インダクタンス [H]
    C : float
        キャパシタンス [F]
    use_spicelib : bool
        SpiceLibを使用するか（デフォルト: True）
    netlist_path : str or Path, optional
        ネットリストファイルのパス

    Returns:
    --------
    Z : ndarray (complex)
        インピーダンス [Ω]
    """
    if use_spicelib and SPICELIB_AVAILABLE:
        try:
            return simulate_rlc_impedance_spicelib(freq, R, L, C, netlist_path)
        except Exception as e:
            print(f"Warning: SpiceLib simulation failed ({e}). Falling back to analytical calculation.")
            return simulate_rlc_impedance_analytical(freq, R, L, C)
    else:
        return simulate_rlc_impedance_analytical(freq, R, L, C)


def objective_function(params, freq, Z_meas, use_spicelib=True, verbose=False):
    """
    目的関数（振幅と位相の重み付き誤差）

    本稿で推奨する方法：
    - 振幅誤差：dBスケールで評価
    - 位相誤差：-180～180度に正規化
    - 重み：振幅10.0、位相1.0

    Parameters:
    -----------
    params : array-like [R, L, C]
        最適化パラメータ
    freq : ndarray
        周波数 [Hz]
    Z_meas : ndarray (complex)
        測定インピーダンス [Ω]
    use_spicelib : bool
        SpiceLibを使用するか
    verbose : bool
        詳細情報を表示するか

    Returns:
    --------
    error : float
        総誤差
    """
    R, L, C = params

    # シミュレーションインピーダンスを計算
    Z_sim = simulate_rlc_impedance(freq, R, L, C, use_spicelib=use_spicelib)

    # 振幅誤差（dBスケール）
    mag_meas_dB = 20 * np.log10(np.abs(Z_meas) + 1e-10)
    mag_sim_dB = 20 * np.log10(np.abs(Z_sim) + 1e-10)
    error_mag = np.sum((mag_meas_dB - mag_sim_dB)**2)

    # 位相誤差（度単位、-180～180度に正規化）
    phase_meas = np.angle(Z_meas, deg=True)
    phase_sim = np.angle(Z_sim, deg=True)
    phase_diff = phase_meas - phase_sim
    phase_diff = np.mod(phase_diff + 180, 360) - 180
    error_phase = np.sum(phase_diff**2) / 180**2

    # 総誤差（重み付け和）
    w_mag = 10.0   # 振幅の重み
    w_phase = 1.0  # 位相の重み
    total_error = w_mag * error_mag + w_phase * error_phase

    if verbose:
        print(f"R={R:.4f}, L={L:.4e}, C={C:.4e}")
        print(f"  Magnitude Error: {error_mag:.4f}")
        print(f"  Phase Error: {error_phase:.4f}")
        print(f"  Total Error: {total_error:.4f}")

    return total_error


def plot_results(freq, Z_meas, Z_sim, title="Impedance Comparison"):
    """
    測定値とシミュレーション結果を比較プロット

    Parameters:
    -----------
    freq : ndarray
        周波数 [Hz]
    Z_meas : ndarray (complex)
        測定インピーダンス [Ω]
    Z_sim : ndarray (complex)
        シミュレーションインピーダンス [Ω]
    title : str
        プロットのタイトル

    Returns:
    --------
    fig : matplotlib.figure.Figure
        図のオブジェクト
    """
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['font.size'] = 14

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 振幅プロット（グレースケール）
    ax1.plot(freq, 20*np.log10(np.abs(Z_meas)), 'k-', label='測定値', linewidth=2)
    ax1.plot(freq, 20*np.log10(np.abs(Z_sim)), color='gray', linestyle='--',
             label='シミュレーション', linewidth=2.5)
    ax1.set_xlabel('周波数 [Hz]', fontsize=16)
    ax1.set_ylabel('|Z| [dB Ω]', fontsize=16)
    ax1.set_title(f'{title} - 振幅特性', fontsize=16)
    ax1.set_xscale('log')
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=13)

    # 位相プロット（グレースケール）
    ax2.plot(freq, np.angle(Z_meas, deg=True), 'k-', label='測定値', linewidth=2)
    ax2.plot(freq, np.angle(Z_sim, deg=True), color='gray', linestyle='--',
             label='シミュレーション', linewidth=2.5)
    ax2.set_xlabel('周波数 [Hz]', fontsize=16)
    ax2.set_ylabel('位相 [度]', fontsize=16)
    ax2.set_title(f'{title} - 位相特性', fontsize=16)
    ax2.set_xscale('log')
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=13)

    plt.tight_layout()

    return fig


def plot_convergence(history, title="Optimization Convergence", best_history=None):
    """
    最適化の収束履歴をプロット

    Parameters:
    -----------
    history : list
        各試行ごとの誤差の履歴
    title : str
        プロットのタイトル
    best_history : list, optional
        最良解の履歴

    Returns:
    --------
    fig : matplotlib.figure.Figure
        図のオブジェクト
    """
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(figsize=(10, 6))

    # 全評価の履歴をプロット（グレースケール）
    ax.plot(history, color='gray', alpha=0.4, linewidth=1, label='全評価')

    # 最良解の履歴をプロット（利用可能な場合）
    if best_history is not None:
        ax.plot(best_history, 'k-', linewidth=2.5, label='最良解')

    ax.set_xlabel('評価回数', fontsize=16)
    ax.set_ylabel('誤差', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)

    plt.tight_layout()

    return fig


def save_results(output_dir, method_name, result, freq, Z_meas, history=None, best_history=None, use_spicelib=True):
    """
    最適化結果を保存

    Parameters:
    -----------
    output_dir : str or Path
        出力ディレクトリ
    method_name : str
        手法名
    result : dict
        最適化結果（'x', 'fun'などを含む辞書）
    freq : ndarray
        周波数 [Hz]
    Z_meas : ndarray (complex)
        測定インピーダンス [Ω]
    history : list, optional
        収束履歴
    best_history : list, optional
        最良解の履歴
    use_spicelib : bool
        SpiceLibを使用したか
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最適パラメータ
    R_opt, L_opt, C_opt = result['x']
    error_final = result['fun']

    # シミュレーション結果を計算
    Z_sim = simulate_rlc_impedance(freq, R_opt, L_opt, C_opt, use_spicelib=use_spicelib)

    # 結果を比較プロット
    fig1 = plot_results(freq, Z_meas, Z_sim, title=f"{method_name} - インピーダンス比較")
    fig1.savefig(output_dir / f"{method_name}_impedance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 収束履歴をプロット
    if history is not None:
        fig2 = plot_convergence(history, title=f"{method_name} - 収束履歴", best_history=best_history)
        fig2.savefig(output_dir / f"{method_name}_convergence.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

    # 結果をテキストファイルに保存
    with open(output_dir / f"{method_name}_results.txt", 'w') as f:
        f.write(f"Optimization Method: {method_name}\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Simulation Method: {'SpiceLib' if use_spicelib and SPICELIB_AVAILABLE else 'Analytical'}\n\n")
        f.write(f"Optimized Parameters:\n")
        f.write(f"  R = {R_opt:.6f} Ω\n")
        f.write(f"  L = {L_opt:.6e} H\n")
        f.write(f"  C = {C_opt:.6e} F\n")
        f.write(f"\nFinal Error: {error_final:.6f}\n")

        if history is not None:
            f.write(f"\nNumber of Iterations: {len(history)}\n")
            f.write(f"Initial Error: {history[0]:.6f}\n")
            f.write(f"Final Error: {history[-1]:.6f}\n")
            f.write(f"Improvement: {(1 - history[-1]/history[0])*100:.2f}%\n")

    # CSVファイルに保存
    results_df = pd.DataFrame({
        'Method': [method_name],
        'R [Ω]': [R_opt],
        'L [H]': [L_opt],
        'C [F]': [C_opt],
        'Final Error': [error_final],
        'Simulation': ['SpiceLib' if use_spicelib and SPICELIB_AVAILABLE else 'Analytical']
    })
    results_df.to_csv(output_dir / f"{method_name}_parameters.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - {method_name}_impedance_comparison.png")
    if history is not None:
        print(f"  - {method_name}_convergence.png")
    print(f"  - {method_name}_results.txt")
    print(f"  - {method_name}_parameters.csv")
    print(f"{'='*60}\n")
