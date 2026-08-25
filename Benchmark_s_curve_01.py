import numpy as np
import time
import os
from matplotlib import pyplot as plt

# Importation directe de vos modules de traitement
try:
    from Cuda_s_curve_01 import script1 as gpu_processor
    from s_curve_01 import script1_cpu as cpu_processor
except ImportError:
    raise ImportError("Assurez-vous que les fichiers 'Cuda_s_curve_01.py' et 's_curve_01.py' sont présents dans le même répertoire.")

"""
Elodees Suite - Outil de Benchmark de Performance Relatif
Compare les performances de traitement par tuiles CPU vs GPU.
Licence: Apache 2.0
"""

def run_benchmark():
    # Génération d'une image de test RAW 3D conséquente (2048 x 2048)
    h_test, w_test = 2048, 2048
    np.random.seed(42)
    img = np.random.rand(h_test, w_test, 3).astype(np.float32)
    
    contrast = 2.5
    t_size = 1024

    print("=" * 60)
    print("      ELODEES PERFORMANCE BENCHMARK - S-CURVE FILTER")
    print("=" * 60)
    print(f"Configuration testée : Matrice RAW 3D ({h_test}x{w_test} | Float32)")
    print(f"Paramètres appliqués : Contraste = {contrast} | Taille de tuile = {t_size}\n")

    # ÉTAPE INDISPENSABLE : Préchauffage (Warm-up) du GPU
    # Permet de compiler le code JIT Numba à blanc pour ne pas fausser les mesures réelles du matériel.
    print("Préchauffage du moteur d'exécution CUDA (Compilation JIT)...")
    _ = gpu_processor(img, contrast_factor=contrast, tile_size=t_size)
    print("Matériel prêt.\n")

    # Mesure des performances : Moteur CPU (NumPy Vectorisé)
    print("Exécution du moteur CPU (NumPy) en cours...")
    start_cpu = time.perf_counter()
    res_cpu = cpu_processor(img, contrast_factor=contrast, tile_size=t_size)
    end_cpu = time.perf_counter()
    duration_cpu = (end_cpu - start_cpu) * 1000  # Conversion en ms

    # Mesure des performances : Moteur GPU (Numba CUDA)
    print("Exécution du moteur GPU (Numba CUDA) en cours...")
    start_gpu = time.perf_counter()
    res_gpu = gpu_processor(img, contrast_factor=contrast, tile_size=t_size)
    end_gpu = time.perf_counter()
    duration_gpu = (end_gpu - start_gpu) * 1000  # Conversion en ms

    # Calcul de l'accélération matérielle brute
    speedup = duration_cpu / duration_gpu

    # --- AFFICHAGE DU TABLEAU DANS LE TERMINAL ---
    print("\n+" + "-" * 56 + "+")
    print(f"| {'PROCESSEUR / MOTEUR':<25} | {'TEMPS DE CALCUL':<24} |")
    print("+" + "-" * 56 + "+")
    print(f"| {'CPU (Intel i7-6700HQ)':<25} | {duration_cpu:>19.2f} ms |")
    print(f"| {'GPU (NVIDIA GTX 980M)':<25} | {duration_gpu:>19.2f} ms |")
    print("+" + "-" * 56 + "+")
    print(f"| FACTEUR D'ACCÉLÉRATION : Le GPU est {speedup:.2f}x plus rapide. |")
    print("+" + "-" * 56 + "+\n")

    # --- GÉNÉRATION DU GRAPHIQUE METRIQUE POUR GITHUB ---
    # Sélection d'un style robuste compatible avec toutes les versions de Matplotlib
    available_styles = plt.style.available
    chosen_style = 'ggplot' if 'ggplot' in available_styles else 'default'
    plt.style.use(chosen_style)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    processors = ['CPU\n(Intel i7-6700HQ)', 'GPU\n(NVIDIA GTX 980M)']
    times = [duration_cpu, duration_gpu]
    colors = ['#34495E', '#2ECC71']  # Teintes sombres élégantes et vert performance

    bars = ax.barh(processors, times, color=colors, height=0.45, edgecolor='none')

    ax.set_title(f"Elodees Core Performance : S-Curve Filter ({h_test}x{w_test} Float32)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Temps d'exécution (millisecondes) - Le plus court est le meilleur", fontsize=10, labelpad=8)
    ax.set_xlim(0, max(times) * 1.18)

    # Intégration des valeurs numériques précises au bout des barres graphiques
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(times) * 0.02), bar.get_y() + bar.get_height()/2,
                f'{width:.1f} ms',
                va='center', ha='left', fontsize=10, fontweight='bold')

    # Boîte d'annotation textuelle mettant en valeur le gain de puissance
    ax.text(max(times) * 0.55, 0.5, f"Speedup: X {speedup:.1f}", 
            fontsize=14, fontweight='bold', color='#2ECC71',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2ECC71', boxstyle='round,pad=0.6'))

    plt.tight_layout()
    
    # Enregistrement du fichier image pour l'intégration automatique dans le README.md
    output_filename = 'benchmark_s_curve.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Graphique métrique enregistré avec succès sous '{output_filename}' 📸")
    plt.show()

if __name__ == "__main__":
    run_benchmark()
