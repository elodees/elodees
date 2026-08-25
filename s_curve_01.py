import numpy as np
import math

"""
Elodees Pipeline - Script 2 (Version CPU vectorisée)
Filtre S-Curve (contraste sigmoïde) par tuiles de données.
Licence: Apache 2.0
"""

def apply_s_curve_cpu_tile(tile_input, k, low, denom):
    """
    Applique la courbe sigmoïde sur une tuile de manière vectorisée via NumPy.
    """
    val = np.clip(tile_input, 0.0001, 0.9999)
    s_val = 1.0 / (1.0 + np.exp(-k * (val - 0.5)))
    return (s_val - low) / denom

def script1_cpu(img, contrast_factor=2.5, tile_size=1024):
    """
    Version CPU équivalente au Script 1 au format de pipeline Elodees.
    Découpe et traite l'image par tuiles pour préserver la mémoire vive (RAM).
    """
    img_float = np.ascontiguousarray(img, dtype=np.float32)
    h, w, _ = img_float.shape
    out_final = np.empty_like(img_float)
    
    k_f32 = np.float32(contrast_factor)
    if abs(k_f32) < 1e-5:
        return img_float
        
    low = np.float32(1.0 / (1.0 + math.exp(k_f32 * 0.5)))
    high = np.float32(1.0 / (1.0 + math.exp(-k_f32 * 0.5)))
    denom = np.float32(high - low)

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile_h, tile_w = min(tile_size, h - i), min(tile_size, w - j)
            tile_cpu = img_float[i:i+tile_h, j:j+tile_w]
            
            # Exécution de l'opération vectorisée
            tile_result = apply_s_curve_cpu_tile(tile_cpu, k_f32, low, denom)
            out_final[i:i+tile_h, j:j+tile_w, :] = tile_result
            
            # Nettoyage de la mémoire pour le garbage collector de Python
            del tile_cpu
            del tile_result

    return out_final

if __name__ == "__main__":
    # Test unitaire autonome du script version CPU
    h_test, w_test = 2048, 2048
    np.random.seed(42)
    img_test = np.random.rand(h_test, w_test, 3).astype(np.float32)

    print("\n--- [CPU] Analyse des métriques de l'image d'entrée ---")
    print(f"Max : {np.max(img_test)} | Min : {np.min(img_test)}")

    result = script1_cpu(img_test, contrast_factor=2.5, tile_size=1024)
    result = np.clip(result, 0.0, 1.0)

    print("\n--- [CPU] Analyse des métriques de l'image de sortie ---")
    print(f"Max : {np.max(result)} | Min : {np.min(result)}")
    print("\nTraitement CPU validé avec succès ✅")
