"""Main program to test inspired sugeno/choquet integrals"""


from fuzzy_integrals import *

if __name__ == "__main__":
    print("--- Inspired Sugeno/Choquet test ---\n")
    
    # Some test data
    x_num = [0.2, 0.8, 0.5]
    n_num = len(x_num)
    
    # Fuzzy measure (cardinality)
    dummy_measure = lambda coalition: len(coalition) / n_num
    
    print("1. Clasic Sugeno:", sugeno_integral(x_num, dummy_measure))
    
    # FG-Functional Sugeno (F=product, G=bounded sum)
    fg_f = lambda a, b: a * b
    fg_g = lambda *args: min(1.0, sum(args))
    print("2. Sugeno FG-Functional:", sugeno_fg_functional(x_num, dummy_measure, fg_f, fg_g))
    
    # Inspired Sugeno (H = average)
    h_func = lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0
    print("3. Sugeno Inspired:", sugeno_inspired_aggregation(x_num, h_func))
    
    # Generalized Inspired Sugeno
    print("4. Generalized Sugeno Inspired (F=prod, G=sum):",
          generalized_sugeno_inspired_aggregation(x_num, lambda a, b: a * b, lambda *args: sum(args), h_func))
    
    print("5. Clasic Choquet:", choquet_integral(x_num, dummy_measure))
    print("6. Choquet Inspired:", choquet_inspired_aggregation(x_num, lambda coalition: np.mean(coalition)))
    
    print("\n--- INTERVAL-VALUED INTEGRALS ---")
    iv_data = [[0.1, 0.3], [0.7, 0.9], [0.4, 0.6]]
    dummy_measure_iv = lambda coalition: [len(coalition) / 3, min(1.0, len(coalition) / 3 + 0.1)]
    h_iv_func = lambda x_hat: [np.mean([iv[0] for iv in x_hat]), np.mean([iv[1] for iv in x_hat])] if len(x_hat) > 0 else [0, 0]
    
    print("7. IV-Clasic Sugeno:", interval_sugeno_integral(iv_data, dummy_measure_iv))
    print("8. IV-Sugeno Inspired:", interval_sugeno_inspired_aggregation(iv_data, h_iv_func))
    print("9. IV-Clasic Choquet:", interval_choquet_integral(iv_data, dummy_measure_iv))
    print("10. IV-Choquet Inspired:", interval_choquet_inspired_aggregation(iv_data, h_iv_func))
