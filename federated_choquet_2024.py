"""
Experimento completo de Aprendizaje Federado (Pękala et al., 2024)
con múltiples repeticiones, tabla resumen detallada y gráficos con barras de error.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from scipy.optimize import brentq

from fuzzy_integrals import sugeno_integral, choquet_integral
from fuzzy_integrals import sugeno_inspired_aggregation, choquet_inspired_aggregation

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0
        
    prc = precision_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
        
    return {"ACC": acc, "SENS": sens, "SPEC": spec, "PRC": prc, "AUC": auc}

def compute_lambda_measure(qualities):
    """Calcula la lambda-medida de Sugeno basada en las calidades Q_i de los clientes."""
    n = len(qualities)
    q = np.array(qualities, dtype=float)
    
    def equation(lam):
        if abs(lam) < 1e-7:
            return 0.0
        return np.prod(1.0 + lam * q) - lam - 1.0
    
    try:
        if abs(np.sum(q) - 1.0) < 1e-5:
            lam = 0.0
        else:
            lam = brentq(equation, -0.99, 100.0)
    except ValueError:
        lam = 0.0

    measure_dict = {(): 0.0}
    for i in range(n):
        measure_dict[(i,)] = q[i]
        
    if n == 3:
        measure_dict[(0, 1)] = q[0] + q[1] + lam * q[0] * q[1]
        measure_dict[(0, 2)] = q[0] + q[2] + lam * q[0] * q[2]
        measure_dict[(1, 2)] = q[1] + q[2] + lam * q[1] * q[2]
        measure_dict[(0, 1, 2)] = 1.0
        
    def fuzzy_measure(coalition):
        sorted_coal = tuple(sorted(coalition))
        return measure_dict.get(sorted_coal, 1.0 if len(sorted_coal) == n else 0.0)
        
    return fuzzy_measure

class IntervalLogisticRegressionSGD:
    """Regresión Logística con SGD adaptada a intervalos mediante el operador Rep_gamma[cite: 4]."""
    def __init__(self, n_features, lr=0.01, gamma=0.5):
        self.lr = lr
        self.gamma = gamma
        self.beta = np.zeros(n_features)
        self.beta_0 = 0.0

    def _rep(self, X_inf, X_sup):
        return X_inf + self.gamma * (X_sup - X_inf)

    def fit_epochs(self, X_inf, X_sup, y, epochs=5):
        n_samples = X_inf.shape[0]
        for _ in range(epochs):
            for i in range(n_samples):
                x_rep = self._rep(X_inf[i], X_sup[i])
                z = self.beta_0 + np.dot(self.beta, x_rep)
                z = np.clip(z, -500, 500)
                pred = 1.0 / (1.0 + np.exp(-z))
                error = pred - y[i]
                self.beta -= self.lr * error * x_rep
                self.beta_0 -= self.lr * error

    def predict_proba(self, X_inf, X_sup):
        x_test = X_inf + 0.5 * (X_sup - X_inf)
        z = self.beta_0 + np.dot(x_test, self.beta)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X_inf, X_sup):
        return (self.predict_proba(X_inf, X_sup) >= 0.5).astype(int)


def run_robust_federated_experiment(scenario_type="iid", num_runs=5, rounds=10):
    print(f"\n=========================================================================================")
    print(f"EXPERIMENTO FEDERADO ROBUSTO ({num_runs} REPETICIONES) - ESCENARIO: {scenario_type.upper()}")
    print(f"=========================================================================================")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    # Construir intervalos [mean - std, mean + std][cite: 4]
    std_approx = np.std(X_norm, axis=0) * 0.1
    X_inf = np.clip(X_norm - std_approx, 0.0, 1.0)
    X_sup = np.clip(X_norm + std_approx, 0.0, 1.0)
    X_inf, X_sup = np.minimum(X_inf, X_sup), np.maximum(X_inf, X_sup)

    model_names = ["Centralized", "Local Avg", "Sugeno FL", "Choquet FL", "Sugeno Inspired FL"]
    metric_keys = ["ACC", "SENS", "SPEC", "PRC", "AUC"]
    runs_results = {m: {k: [] for k in metric_keys} for m in model_names}
    
    for run in range(num_runs):
        seed = 42 + run
        
        # 1. Particionado
        if scenario_type == "iid":
            idx_c1, idx_rem = train_test_split(np.arange(len(y)), train_size=1/3, stratify=y, random_state=seed)
            idx_c2, idx_c3 = train_test_split(idx_rem, train_size=0.5, stratify=y[idx_rem], random_state=seed)
        else:
            idx_pos = np.where(y == 1)[0]
            idx_neg = np.where(y == 0)[0]
            np.random.seed(seed)
            np.random.shuffle(idx_pos)
            np.random.shuffle(idx_neg)
            idx_c1 = np.concatenate([idx_pos[:70], idx_neg[:70]])
            idx_c2 = np.concatenate([idx_pos[70:200], idx_neg[70:100]])
            idx_c3 = np.concatenate([idx_pos[200:], idx_neg[100:]])

        clients_indices = [idx_c1, idx_c2, idx_c3]
        _, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        X_test_inf, X_test_sup, y_test = X_inf[idx_test], X_sup[idx_test], y[idx_test]
        X_train_inf, X_train_sup, y_train, _ = X_inf[~np.isin(np.arange(len(y)), idx_test)], X_sup[~np.isin(np.arange(len(y)), idx_test)], y[~np.isin(np.arange(len(y)), idx_test)], y[~np.isin(np.arange(len(y)), idx_test)]

        # --- MODELO 1: Centralizado ---
        cent_model = LogisticRegression(max_iter=1000).fit(X_train_inf + 0.5*(X_train_sup - X_train_inf), y_train)
        y_p = cent_model.predict(X_test_inf + 0.5*(X_test_sup - X_test_inf))
        y_prob = cent_model.predict_proba(X_test_inf + 0.5*(X_test_sup - X_test_inf))[:, 1]
        res_cent = compute_metrics(y_test, y_p, y_prob)
        for k in metric_keys:
            runs_results["Centralized"][k].append(res_cent[k])

        # --- MODELO 2: Local Avg ---
        local_metrics_run = {k: [] for k in metric_keys}
        for indices in clients_indices:
            loc_model = LogisticRegression(max_iter=1000).fit(X_inf[indices] + 0.5*(X_sup[indices] - X_inf[indices]), y[indices])
            lp = loc_model.predict(X_test_inf + 0.5*(X_test_sup - X_test_inf))
            lpr = loc_model.predict_proba(X_test_inf + 0.5*(X_test_sup - X_test_inf))[:, 1]
            lm = compute_metrics(y_test, lp, lpr)
            for k in metric_keys:
                local_metrics_run[k].append(lm[k])
        for k in metric_keys:
            runs_results["Local Avg"][k].append(np.mean(local_metrics_run[k]))

        # --- MODELOS 3 & 4: Sugeno FL y Choquet FL (Bucle Iterativo) ---
        for agg_name, agg_func in [("Sugeno FL", sugeno_integral), ("Choquet FL", choquet_integral)]:
            n_features = X_inf.shape[1]
            # initial params send by the server (betas)
            global_beta = np.zeros(n_features)
            global_beta_0 = 0.0
            local_models = [IntervalLogisticRegressionSGD(n_features) for _ in range(3)]

            # How many times the clientes send the data and receive the aggregated model
            for _ in range(rounds):
                local_accuracies = []
                client_betas, client_beta0s = [], []

                for c_idx, indices in enumerate(clients_indices):
                    local_models[c_idx].beta = np.copy(global_beta)
                    local_models[c_idx].beta_0 = np.copy(global_beta_0)
                    local_models[c_idx].fit_epochs(X_inf[indices], X_sup[indices], y[indices], epochs=3)
                    
                    preds = local_models[c_idx].predict(X_inf[indices], X_sup[indices])
                    acc_loc = accuracy_score(y[indices], preds)
                    local_accuracies.append(max(0.01, min(0.99, acc_loc)))
                    
                    client_betas.append(local_models[c_idx].beta)
                    client_beta0s.append(local_models[c_idx].beta_0)

                fuzzy_measure = compute_lambda_measure(local_accuracies)
                beta_matrix = np.array(client_betas).T

                # Data aggregation
                new_beta = np.array([agg_func(beta_matrix[j], fuzzy_measure) for j in range(n_features)])
                new_beta_0 = agg_func(np.array(client_beta0s), fuzzy_measure)

                global_beta, global_beta_0 = new_beta, new_beta_0

            final_fl = IntervalLogisticRegressionSGD(n_features)
            final_fl.beta, final_fl.beta_0 = global_beta, global_beta_0
            fl_pred = final_fl.predict(X_test_inf, X_test_sup)
            fl_prob = final_fl.predict_proba(X_test_inf, X_test_sup)
            res_fl = compute_metrics(y_test, fl_pred, fl_prob)
            for k in metric_keys:
                runs_results[agg_name][k].append(res_fl[k])
            
        # Definimos la función H inspirada (ej: media de la coalición restante x_hat)
        h_inspired_func = lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0

        # --- MODELO 5: Sugeno Inspired FL ---
        n_features = X_inf.shape[1]
        # initial params send by the server (betas)
        global_beta_insp = np.zeros(n_features)
        global_beta_0_insp = 0.0
        local_models_insp = [IntervalLogisticRegressionSGD(n_features) for _ in range(3)]

        # How many times the clientes send the data and receive the aggregated model
        for _ in range(rounds):
            client_betas_insp, client_beta0s_insp = [], []
            for c_idx, indices in enumerate(clients_indices):
                local_models_insp[c_idx].beta = np.copy(global_beta_insp)
                local_models_insp[c_idx].beta_0 = np.copy(global_beta_0_insp)
                local_models_insp[c_idx].fit_epochs(X_inf[indices], X_sup[indices], y[indices], epochs=3)
                
                client_betas_insp.append(local_models_insp[c_idx].beta)
                client_beta0s_insp.append(local_models_insp[c_idx].beta_0)

            beta_matrix_insp = np.array(client_betas_insp).T
            
            # Agregación usando la función inspirada de Sugeno
            new_beta_insp = np.array([
                sugeno_inspired_aggregation(beta_matrix_insp[j], h_inspired_func) 
                for j in range(n_features)
            ])
            new_beta_0_insp = sugeno_inspired_aggregation(np.array(client_beta0s_insp), h_inspired_func)

            global_beta_insp, global_beta_0_insp = new_beta_insp, new_beta_0_insp

        # Evaluar y guardar en el diccionario de resultados
        final_insp = IntervalLogisticRegressionSGD(n_features)
        final_insp.beta, final_insp.beta_0 = global_beta_insp, global_beta_0_insp
        res_insp = compute_metrics(y_test, final_insp.predict(X_test_inf, X_test_sup), final_insp.predict_proba(X_test_inf, X_test_sup))
        for k in metric_keys: 
            runs_results["Sugeno Inspired FL"][k].append(res_insp[k])

    # Imprimir tabla resumen detallada en consola
    print(f"\nRESULTADOS FINALES ({num_runs} REPETICIONES) - {scenario_type.upper()}")
    header = f"{'Modelo':<15} | " + " | ".join([f"{k:<18}" for k in metric_keys])
    print(header)
    print("-" * len(header))
    
    for model in model_names:
        row = f"{model:<15} | "
        for k in metric_keys:
            mean_val = np.mean(runs_results[model][k])
            std_val = np.std(runs_results[model][k])
            row += f"{mean_val:.3f} ± {std_val:.3f}   | "
        print(row)

    # Generar y guardar gráfico
    plot_results(runs_results, model_names, metric_keys, scenario_type)

def plot_results(runs_results, model_names, metric_keys, scenario_type):
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)  # Dinámico según el número de modelos
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Paleta ampliada y moderna con suficientes colores para todos los modelos
    colors = ['#2b5c8f', '#4682b4', '#2e8b57', '#e67e22', '#9b59b6', '#e74c3c']
    
    for i, model in enumerate(model_names):
        means = [np.mean(runs_results[model][k]) for k in metric_keys]
        stds = [np.std(runs_results[model][k]) for k in metric_keys]
        
        # Cálculo del desplazamiento centrado para cualquier número de modelos
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, 
               label=model, color=colors[i % len(colors)], alpha=0.85)
        
    ax.set_ylabel('Valor de la Métrica', fontsize=12)
    ax.set_title(f'Rendimiento Comparativo - Escenario {scenario_type.upper()} ({len(runs_results[model_names[0]]["ACC"])} repeticiones)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filename = f"federated_results_{scenario_type}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n[Gráfico guardado como: {filename}]")
    plt.show()

if __name__ == "__main__":
    run_robust_federated_experiment(scenario_type="iid", num_runs=5, rounds=5)
    run_robust_federated_experiment(scenario_type="non-iid", num_runs=5, rounds=5)
