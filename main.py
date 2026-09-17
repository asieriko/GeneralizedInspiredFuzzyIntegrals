from functools import partial
from federated_LogisticRegression_2024 import run_robust_federated_experiment
from fuzzy_integrals import *

if __name__ == "__main__":
    # H function for the inspired integral (ej. average of the remaining coaliton)
    h_inspired_func = lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0

    # Functions
    aggregation_methods = {
        "Sugeno FL": partial(sugeno_integral),  # Requires (vector, fuzzy_measure)
        "Choquet FL": partial(choquet_integral),  # Requires (vector, fuzzy_measure)
        "Sugeno Inspired FL": partial(sugeno_inspired_aggregation, H=h_inspired_func)
        # Requires (vector, h_func)
    }


    run_robust_federated_experiment(aggregation_methods, scenario_type="iid", num_runs=5, rounds=5)
    run_robust_federated_experiment(aggregation_methods, scenario_type="non-iid", num_runs=5, rounds=5)
