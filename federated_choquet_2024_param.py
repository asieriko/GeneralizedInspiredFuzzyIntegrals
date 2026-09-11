import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from scipy.optimize import brentq

# Importamos las funciones numéricas y de intervalos de nuestra librería
from fuzzy_integrals import (
    sugeno_integral, 
    choquet_integral, 
    sugeno_inspired_aggregation, 
    choquet_inspired_aggregation,
    interval_choquet_integral,
    generalized_sugeno_inspired_aggregation
)

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0.0
    prc = precision_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    return {"ACC": acc, "SENS": sens, "SPEC": spec, "PRC": prc, "AUC": auc}

def compute_lambda_measure(qualities):
    n = len(qualities)
    q = np.array(qualities, dtype=float)
    def equation(lam):
        if abs(lam) < 1e-7: return 0.0
        return np.prod(1.0 + lam * q) - lam - 1.0
    try:
        lam = 0.0 if abs(np.sum(q) - 1.0) < 1e-5 else brentq(equation, -0.99, 100.0)
    except ValueError:
        lam = 0.0

    measure_dict = {(): 0.0}
    for i in range(n): measure_dict[(i,)] = q[i]
    if n == 3:
        measure_dict[(0, 1)] = q[0] + q[1] + lam * q[0] * q[1]
        measure_dict[(0, 2)] = q[0] + q[2] + lam * q[0] * q[2]
        measure_dict[(1, 2)] = q[1] + q[2] + lam * q[1] * q[2]
        measure_dict[(0, 1, 2)] = 1.0
        
    return lambda coalition: measure_dict.get(tuple(sorted(coalition)), 1.0 if len(coalition) == n else 0.0)

def compute_lambda_measure_iv(qualities):
    """Genera una medida difusa en intervalos basada en la lambda-medida real."""
    real_m = compute_lambda_measure(qualities)
    def iv_measure(coalition):
        val = real_m(coalition)
        return [max(0.0, val - 0.05), min(1.0, val + 0.05)]
    return iv_measure

class IntervalLogisticRegressionSGD:
    def __init__(self, n_features, lr=0.01, gamma=0.5):
        self.lr = lr
        self.gamma = gamma
        self.beta = np.zeros(n_features)
        self.beta_0 = 0.0

    def fit_epochs(self, X_inf, X_sup, y, epochs=3):
        n_samples = X_inf.shape[0]
        for _ in range(epochs):
            for i in range(n_samples):
                x_rep = X_inf[i] + self.gamma * (X_sup[i] - X_inf[i])
                z = np.clip(self.beta_0 + np.dot(self.beta, x_rep), -500, 500)
                pred = 1.0 / (1.0 + np.exp(-z))
                error = pred - y[i]
                self.beta -= self.lr * error * x_rep
                self.beta_0 -= self.lr * error

    def predict_proba(self, X_inf, X_sup):
        x_test = X_inf + 0.5 * (X_sup - X_inf)
        z = np.clip(self.beta_0 + np.dot(x_test, self.beta), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X_inf, X_sup):
        return (self.predict_proba(X_inf, X_sup) >= 0.5).astype(int)


def run_robust_federated_experiment(scenario_type="iid", num_runs=5, rounds=5, aggregation_methods=None):
    print(f"\n=========================================================================================")
    print(f"EXPERIMENTO PARAMETRIZADO ({num_runs} REPETICIONES) - ESCENARIO: {scenario_type.upper()}")
    print(f"=========================================================================================")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    std_approx = np.std(X_norm, axis=0) * 0.1
    X_inf = np.clip(X_norm - std_approx, 0.0, 1.0)
    X_sup = np.clip(X_norm + std_approx, 0.0, 1.0)

    metric_keys = ["ACC", "SENS", "SPEC", "PRC", "AUC"]
    model_names = [m["name"] for m in aggregation_methods]
    runs_results = {m: {k: [] for k in metric_keys} for m in model_names}
    
    for run in range(num_runs):
        seed = 42 + run
        if scenario_type == "iid":
            idx_c1, idx_rem = train_test_split(np.arange(len(y)), train_size=1/3, stratify=y, random_state=seed)
            idx_c2, idx_c3 = train_test_split(idx_rem, train_size=0.5, stratify=y[idx_rem], random_state=seed)
        else:
            idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
            np.random.seed(seed); np.random.shuffle(idx_pos); np.random.shuffle(idx_neg)
            idx_c1 = np.concatenate([idx_pos[:70], idx_neg[:70]])
            idx_c2 = np.concatenate([idx_pos[70:200], idx_neg[70:100]])
            idx_c3 = np.concatenate([idx_pos[200:], idx_neg[100:]])

        clients_indices = [idx_c1, idx_c2, idx_c3]
        _, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        X_test_inf, X_test_sup, y_test = X_inf[idx_test], X_sup[idx_test], y[idx_test]
        X_train_inf, X_train_sup, y_train = X_inf[~np.isin(np.arange(len(y)), idx_test)], X_sup[~np.isin(np.arange(len(y)), idx_test)], y[~np.isin(np.arange(len(y)), idx_test)]

        # 1. Centralizado
        cent_model = LogisticRegression(max_iter=1000).fit(X_train_inf + 0.5*(X_train_sup - X_train_inf), y_train)
        res_cent = compute_metrics(y_test, cent_model.predict(X_test_inf + 0.5*(X_test_sup - X_test_inf)), cent_model.predict_proba(X_test_inf + 0.5*(X_test_sup - X_test_inf))[:, 1])
        for k in metric_keys: runs_results["Centralized"][k].append(res_cent[k])

        # 2. Local Avg
        local_metrics_run = {k: [] for k in metric_keys}
        for indices in clients_indices:
            loc_model = LogisticRegression(max_iter=1000).fit(X_inf[indices] + 0.5*(X_sup[indices] - X_inf[indices]), y[indices])
            lm = compute_metrics(y_test, loc_model.predict(X_test_inf + 0.5*(X_test_sup - X_test_inf)), loc_model.predict_proba(X_test_inf + 0.5*(X_test_sup - X_test_inf))[:, 1])
            for k in metric_keys: local_metrics_run[k].append(lm[k])
        for k in metric_keys: runs_results["Local Avg"][k].append(np.mean(local_metrics_run[k]))

        # 3. Métodos Federados Parametrizados (Medida, Inspirados e Intervalos)
        for method in aggregation_methods:
            if method["name"] in ["Centralized", "Local Avg"]: continue
            
            n_features = X_inf.shape[1]
            global_beta = np.zeros(n_features)
            global_beta_0 = 0.0
            local_models = [IntervalLogisticRegressionSGD(n_features) for _ in range(3)]

            for _ in range(rounds):
                local_accuracies, client_betas, client_beta0s = [], [], []
                for c_idx, indices in enumerate(clients_indices):
                    local_models[c_idx].beta = np.copy(global_beta)
                    local_models[c_idx].beta_0 = np.copy(global_beta_0)
                    local_models[c_idx].fit_epochs(X_inf[indices], X_sup[indices], y[indices], epochs=3)
                    
                    acc_loc = accuracy_score(y[indices], local_models[c_idx].predict(X_inf[indices], X_sup[indices]))
                    local_accuracies.append(max(0.01, min(0.99, acc_loc)))
                    client_betas.append(local_models[c_idx].beta)
                    client_beta0s.append(local_models[c_idx].beta_0)

                beta_matrix = np.array(client_betas).T
                
                if method["type"] == "measure":
                    fuzzy_measure = compute_lambda_measure(local_accuracies)
                    new_beta = np.array([method["func"](beta_matrix[j], fuzzy_measure) for j in range(n_features)])
                    new_beta_0 = method["func"](np.array(client_beta0s), fuzzy_measure)
                elif method["type"] == "inspired":
                    h_func = method["H"]
                    new_beta = np.array([method["func"](beta_matrix[j], h_func) for j in range(n_features)])
                    new_beta_0 = method["func"](np.array(client_beta0s), h_func)
                elif method["type"] == "interval":
                    iv_measure = compute_lambda_measure_iv(local_accuracies)
                    new_beta = np.zeros(n_features)
                    for j in range(n_features):
                        iv_input = [[val, val] for val in beta_matrix[j]]
                        res_iv = method["func"](iv_input, iv_measure)
                        new_beta[j] = (res_iv[0] + res_iv[1]) / 2.0
                    iv_input_0 = [[val, val] for val in client_beta0s]
                    res_iv_0 = method["func"](iv_input_0, iv_measure)
                    new_beta_0 = (res_iv_0[0] + res_iv_0[1]) / 2.0
                elif method["type"] == "generalized_inspired":
                    h_func = method["H"]
                    F_arg = method["F"]
                    G_arg = method["G"]
                    new_beta = np.array([method["func"](beta_matrix[j], F_arg, G_arg, h_func) for j in range(n_features)])
                    new_beta_0 = method["func"](np.array(client_beta0s), F_arg, G_arg, h_func)

                global_beta, global_beta_0 = new_beta, new_beta_0

            final_fl = IntervalLogisticRegressionSGD(n_features)
            final_fl.beta, final_fl.beta_0 = global_beta, global_beta_0
            res_fl = compute_metrics(y_test, final_fl.predict(X_test_inf, X_test_sup), final_fl.predict_proba(X_test_inf, X_test_sup))
            for k in metric_keys: runs_results[method["name"]][k].append(res_fl[k])

    # Imprimir tabla resumen
    print(f"\nRESULTADOS FINALES ({num_runs} REPETICIONES) - {scenario_type.upper()}")
    header = f"{'Modelo':<24} | " + " | ".join([f"{k:<18}" for k in metric_keys])
    print(header); print("-" * len(header))
    for model in model_names:
        row = f"{model:<24} | "
        for k in metric_keys:
            m_val, s_val = np.mean(runs_results[model][k]), np.std(runs_results[model][k])
            row += f"{m_val:.3f} ± {s_val:.3f}   | "
        print(row)

    plot_results(runs_results, model_names, metric_keys, scenario_type)

def plot_results(runs_results, model_names, metric_keys, scenario_type):
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # GENERACIÓN DINÁMICA DE COLORES: Usa un mapa de colores adaptativo según la cantidad de modelos
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / max(1, len(model_names) - 1)) for i in range(len(model_names))]
    
    for i, model in enumerate(model_names):
        means = [np.mean(runs_results[model][k]) for k in metric_keys]
        stds = [np.std(runs_results[model][k]) for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model, color=colors[i], alpha=0.85)
        
    ax.set_ylabel('Valor de la Métrica', fontsize=12)
    ax.set_title(f'Rendimiento Comparativo - Escenario {scenario_type.upper()} ({len(runs_results[model_names[0]]["ACC"])} repeticiones)', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylim(0.0, 1.05); ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"federated_results_{scenario_type}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Lista parametrizada incluyendo modelos clásicos, inspirados y de intervalos
    custom_aggregations = [
        {"name": "Centralized", "type": "baseline"},
        {"name": "Local Avg", "type": "baseline"},
        {"name": "Sugeno FL", "type": "measure", "func": sugeno_integral},
        {"name": "Choquet FL", "type": "measure", "func": choquet_integral},
        # {"name": "Sugeno Inspired FL", "type": "inspired", "func": sugeno_inspired_aggregation, "H": lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0},
        # {"name": "Interval Choquet FL", "type": "interval", "func": interval_choquet_integral},
        # {"name": "Sugeno Gen Custom", "type": "generalized_inspired",  "func": generalized_sugeno_inspired_aggregation,\
        #  "F": lambda a, b: min(a, b), "G": lambda *args: max(args), "H": lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0}
    ]

    run_robust_federated_experiment(scenario_type="iid", num_runs=3, rounds=3, aggregation_methods=custom_aggregations)
    run_robust_federated_experiment(scenario_type="non-iid", num_runs=3, rounds=3, aggregation_methods=custom_aggregations)
