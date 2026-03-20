"""
QWKR - Quantum Wasserstein Kernel Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_density_matrices():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    rho_full = []
    rho_partial = []

    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        rho = sv.to_operator().data
        rho_full.append(rho)

        trace_out = list(range(2, NUM_QUBITS))
        rho_p = partial_trace(sv, trace_out).data
        rho_partial.append(rho_p)

    return rho_full, rho_partial


def wasserstein_kernel(rho_list):
    n = len(rho_list)
    K = np.zeros((n, n))

    eigenvals = []
    for rho in rho_list:
        vals = np.linalg.eigvalsh(rho)
        vals = np.sort(vals)[::-1]
        eigenvals.append(vals)

    for i in range(n):
        for j in range(i, n):
            diff = eigenvals[i] - eigenvals[j]
            dist = np.sqrt(np.sum(diff ** 2))
            K[i, j] = np.exp(-dist)
            K[j, i] = K[i, j]

    return K


def fidelity_kernel(rho_list):
    n = len(rho_list)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            prod = rho_list[i] @ rho_list[j]
            fid = np.real(np.trace(prod))
            K[i, j] = fid
            K[j, i] = fid

    return K


def ridge_predict(K, y, lam=LAMBDA_REG):
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return K @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantne matrice gustine ({NUM_QUBITS}q) ---")
    rho_full, rho_partial = compute_density_matrices()
    print(f"  Pune rho: {len(rho_full)}x{rho_full[0].shape}")
    print(f"  Parcijalne rho (trace qubiti 2-4): "
          f"{len(rho_partial)}x{rho_partial[0].shape}")

    print(f"\n--- Racunanje kernela ---")
    K_wass = wasserstein_kernel(rho_full)
    print(f"  Wasserstein kernel: rang={np.linalg.matrix_rank(K_wass)}")

    K_fid_full = fidelity_kernel(rho_full)
    print(f"  Fidelity kernel (full): rang={np.linalg.matrix_rank(K_fid_full)}")

    K_fid_part = fidelity_kernel(rho_partial)
    print(f"  Fidelity kernel (partial): rang={np.linalg.matrix_rank(K_fid_part)}")

    K_combined = (K_wass + K_fid_full + K_fid_part) / 3.0
    print(f"  Kombinovani kernel: rang={np.linalg.matrix_rank(K_combined)}")

    print(f"\n--- QWKR po pozicijama ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = ridge_predict(K_combined, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QWKR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantne matrice gustine (5q) ---
  Pune rho: 32x(32, 32)
  Parcijalne rho (trace qubiti 2-4): 32x(4, 4)

--- Racunanje kernela ---
  Wasserstein kernel: rang=1
  Fidelity kernel (full): rang=32
  Fidelity kernel (partial): rang=13
  Kombinovani kernel: rang=32

--- QWKR po pozicijama ---
  Poz 1 [1-33]: 1:0.167 | 2:0.146 | 3:0.129
  Poz 2 [2-34]: 8:0.086 | 5:0.076 | 9:0.076
  Poz 3 [3-35]: 13:0.064 | 12:0.063 | 14:0.062
  Poz 4 [4-36]: 23:0.064 | 21:0.063 | 18:0.063
  Poz 5 [5-37]: 29:0.065 | 26:0.063 | 27:0.063
  Poz 6 [6-38]: 33:0.083 | 32:0.081 | 35:0.080
  Poz 7 [7-39]: 7:0.182 | 38:0.153 | 37:0.132

==================================================
Predikcija (QWKR, deterministicki, seed=39):
[1, 8, x, y, z, 33, 38]
==================================================
"""



"""
QWKR - Quantum Wasserstein Kernel Regression

Radi sa matricama gustine (density matrices) umesto statevectors/fidelity
3 kernela kombinovano:
Wasserstein kernel: udaljenost izmedju spektara (eigenvalues) matrica gustine - geometrijska mera
Fidelity kernel (full): Tr(rho_i @ rho_j) na punoj 5-qubit matrici
Fidelity kernel (partial): isti ali na parcijalnom tragu (trace out qubiti 2-4) - hvata lokalne korelacije prvih 2 qubita
partial_trace iz Qiskit-a za redukovane matrice gustine
Deterministicki, bez treniranja kola
"""

