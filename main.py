"""programa principal para probar sugeno/choquet inspired"""


import numpy as np
from fuzzy_integrals import *

# ==========================================
# 4. EJEMPLO DE USO Y PRUEBAS
# ==========================================
if __name__ == "__main__":
    print("--- PRUEBAS DE AGREGACIÓN AVANZADA ---\n")
    
    # Datos de prueba numéricos
    x_num = [0.2, 0.8, 0.5]
    n_num = len(x_num)
    
    # Medida difusa simulada (Cardinalidad normalizada)
    dummy_measure = lambda coalition: len(coalition) / n_num
    
    print("1. Sugeno Clásico:", sugeno_integral(x_num, dummy_measure))
    
    # FG-Functional de Sugeno (F=producto, G=suma acotada)
    fg_f = lambda a, b: a * b
    fg_g = lambda *args: min(1.0, sum(args))
    print("2. Sugeno FG-Functional:", sugeno_fg_functional(x_num, dummy_measure, fg_f, fg_g))
    
    # Sugeno Inspirada (H = media de los restantes)
    h_func = lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0
    print("3. Sugeno Inspirada:", sugeno_inspired_aggregation(x_num, h_func))
    
    # Sugeno Inspirada Generalizada
    print("4. Sugeno Inspirada Gen (F=prod, G=suma):", 
          generalized_sugeno_inspired_aggregation(x_num, lambda a, b: a * b, lambda *args: sum(args), h_func))
    
    print("5. Choquet Clásico:", choquet_integral(x_num, dummy_measure))
    print("6. Choquet Inspirado:", choquet_inspired_aggregation(x_num, lambda coalition: np.mean(coalition)))
    
    print("\n--- PRUEBAS DE INTERVALOS (INTERVAL-VALUED) ---")
    iv_data = [[0.1, 0.3], [0.7, 0.9], [0.4, 0.6]]
    dummy_measure_iv = lambda coalition: [len(coalition) / 3, min(1.0, len(coalition) / 3 + 0.1)]
    h_iv_func = lambda x_hat: [np.mean([iv[0] for iv in x_hat]), np.mean([iv[1] for iv in x_hat])] if len(x_hat) > 0 else [0, 0]
    
    print("7. IV-Sugeno Clásico:", interval_sugeno_integral(iv_data, dummy_measure_iv))
    print("8. IV-Sugeno Inspirado:", interval_sugeno_inspired_aggregation(iv_data, h_iv_func))
    print("9. IV-Choquet Clásico:", interval_choquet_integral(iv_data, dummy_measure_iv))
    print("10. IV-Choquet Inspirado:", interval_choquet_inspired_aggregation(iv_data, h_iv_func))
