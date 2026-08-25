import numpy as np
import math
from numba import cuda

"""
Elodees Pipeline - Script 1 (Version CUDA GPU)
Filtre S-Curve (contraste sigmoïde) optimisé par tuiles dynamiques.
Licence: Apache 2.0
"""

@cuda.jit
def s_curve_kernel(img, k, low, denom, out):
    """
    Kernel CUDA optimisé. Les constantes d'échelle (low, denom) sont 
    pré-calculées sur le CPU pour économiser les ressources de calcul de l'ALU.
    """
    row, col = cuda.grid(2)
    h, w, c_count = img.shape
    
    if row < h and col < w:
        for c in range(3):
            val = img[row, col, c]
            
            # Clamp de sécurité pour éviter l'instabilité de la fonction exponentielle
            val = max(0.0001, min(0.9999, val))
            
            # Application de la courbe sigmoïde centrée sur 0.5
            s_val = 1.0 / (1.0 + math.exp(-k * (val - 0.5)))
            
            # Normalisation finale sur la plage dynamique [0, 1]
            out[row, col, c] = (s_val - low) / denom

def script1(img, contrast_factor=2.5, tile_size=1024):
    """
    Fonction principale du pipeline Elodees. 
    Découpe l'image et orchestre le traitement par tuiles matérielles GPU.
    """
    # Garantie de l'alignement mémoire float32 pour des transferts DMA optimaux
    img_float = np.ascontiguousarray(img, dtype=np.float32)
    h, w, _ = img_float.shape
    out_final = np.empty_like(img_float)
    
    k_f32 = np.float32(contrast_factor)
    if abs(k_f32) < 1e-5:
        return img_float
        
    low = np.float32(1.0 / (1.0 + math.exp(k_f32 * 0.5)))
    high = np.float32(1.0 / (1.0 + math.exp(-k_f32 * 0.5)))
    denom = np.float32(high - low)
    
    # 256 threads par bloc : configuration idéale pour l'architecture Maxwell de la GTX 980M
    threads_per_block = (16, 16)

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile_h, tile_w = min(tile_size, h - i), min(tile_size, w - j)
            
            # Extraction de la tuile contiguë en RAM
            tile_cpu = np.ascontiguousarray(img_float[i:i+tile_h, j:j+tile_w])
            
            # Allocation et transfert vers la VRAM
            d_src = cuda.to_device(tile_cpu)
            d_dst = cuda.device_array(tile_cpu.shape, dtype=np.float32)
            
            # Grille de calcul ajustée à la géométrie stricte de la tuile courante
            blocks_per_grid = (
                math.ceil(tile_h / threads_per_block[0]), 
                math.ceil(tile_w / threads_per_block[1])
            )
            
            # Exécution du kernel
            s_curve_kernel[blocks_per_grid, threads_per_block](d_src, k_f32, low, denom, d_dst)
            
            # Barrière de synchronisation par tuile
            cuda.synchronize()
            
            # Rapatriement des données calculées vers la RAM hôte
            out_final[i:i+tile_h, j:j+tile_w, :] = d_dst.copy_to_host()
            
            # Libération immédiate et explicite des pointeurs de la VRAM
            del d_src
            del d_dst
            
            # Nettoyage forcé du contexte mémoire CUDA pour la tuile suivante
            with cuda.defer_cleanup():
                pass

    # Synchronisation finale de sécurité avant le retour du flux
    cuda.synchronize()
    return out_final

if __name__ == "__main__":
    # Test unitaire autonome du script 1
    h_test, w_test = 2048, 2048
    np.random.seed(42)
    img_test = np.random.rand(h_test, w_test, 3).astype(np.float32)

    print("\n--- [GPU] Analyse des métriques de l'image d'entrée ---")
    print(f"Max : {np.max(img_test)} | Min : {np.min(img_test)} | Moyenne : {np.mean(img_test)}")

    # Appel au format de votre pipeline séquentiel
    result = script1(img_test, contrast_factor=2.5, tile_size=1024)
    result = np.clip(result, 0.0, 1.0)

    print("\n--- [GPU] Analyse des métriques de l'image de sortie ---")
    print(f"Max : {np.max(result)} | Min : {np.min(result)} | Moyenne : {np.mean(result)}")
    print("\nTraitement GPU validé avec succès ✅")
