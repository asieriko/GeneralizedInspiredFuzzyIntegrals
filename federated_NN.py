import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Neural Network Architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, num_classes=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. Traditional Aggregation: Federated Averaging (FedAvg)
def aggregate_fedavg(global_model, client_models, data_weights):
    """
    Standard FedAvg based on the volume of data or uniform weights.
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = sum(
            client_models[i].state_dict()[k] * data_weights[i]
            for i in range(len(client_models))
        )
    global_model.load_state_dict(global_dict)
    return global_model


# 3. Advanced Aggregation: Sugeno, Choquet, and OWA Integrals
def aggregate_integrals(global_model, client_models, client_qualities, method='sugeno'):
    """
    Aggregates neural network weights using fuzzy integrals (Sugeno, Choquet)
    or the Ordered Weighted Averaging (OWA) operator.
    Requires parameters normalized to [0,1].
    """
    global_dict = global_model.state_dict()

    # Q_i represents the local model quality (e.g., accuracy)
    Q = torch.tensor(client_qualities, dtype=torch.float32)

    for k in global_dict.keys():
        # Stack parameters from all clients: shape (num_clients, *param_shape)
        stacked_params = torch.stack([m.state_dict()[k] for m in client_models], dim=0)

        # Step A: Normalization to [0, 1] range for fuzzy measure compatibility
        p_min = stacked_params.min(dim=0, keepdim=True)[0]
        p_max = stacked_params.max(dim=0, keepdim=True)[0]
        eps = 1e-9
        normalized_params = (stacked_params - p_min) / (p_max - p_min + eps)

        # Step B: Descending sort h(x_1) >= h(x_2) >= ... >= h(x_n)
        sorted_vals, sorted_indices = torch.sort(normalized_params, dim=0, descending=True)

        # Step C: Fuzzy measure m(A_i) based on cumulative client qualities
        client_measures = Q[sorted_indices]
        m_A = torch.cumsum(client_measures, dim=0)
        m_A = m_A / m_A[-1:]  # Normalize measure to max 1.0

        # Step D: Apply the specific Integral formulation
        if method == 'sugeno':
            # Sm(x) = max_i ( min( h(x_i), m(A_i) ) )
            min_tensor = torch.min(sorted_vals, m_A)
            aggregated_norm = torch.max(min_tensor, dim=0)[0]

        elif method == 'choquet':
            # Cm(x) = sum ( m(A_i) * (h(x_i) - h(x_{i+1})) )
            zeros_shift = torch.zeros((1, *sorted_vals.shape[1:]), device=sorted_vals.device)
            shifted_vals = torch.cat([sorted_vals[1:], zeros_shift], dim=0)
            differences = sorted_vals - shifted_vals
            aggregated_norm = torch.sum(m_A * differences, dim=0)

        elif method == 'owa':
            # Ordered Weighted Averaging (OWA)
            # Uses predefined descending weights for the sorted parameters
            n = len(client_models)
            owa_weights = torch.tensor([(2 * (n - i) - 1) / (n ** 2) for i in range(n)], dtype=torch.float32)
            owa_weights = owa_weights.view(-1, *([1] * (sorted_vals.dim() - 1)))
            aggregated_norm = torch.sum(owa_weights * sorted_vals, dim=0)

        # Step E: Denormalize back to the original weight scale
        aggregated_params = aggregated_norm * (p_max.squeeze(0) - p_min.squeeze(0) + eps) + p_min.squeeze(0)
        global_dict[k] = aggregated_params

    global_model.load_state_dict(global_dict)
    return global_model


# 4. Evaluation and Plotting
def evaluate_model(model, X_test, y_test):
    """
    Returns the accuracy metric for a given model.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(y_test.numpy(), predicted.numpy())
    return acc


def plot_metrics(results_dict):
    """
    Generates a bar chart comparing the aggregation strategies.
    """
    labels = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1.1)
    plt.ylabel('Global Model Accuracy')
    plt.title('Comparison of FL Aggregation Strategies (Neural Networks)')

    # Add metric values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

    plt.show()


# 5. Simulation Execution
if __name__ == "__main__":
    # Generate synthetic dataset and split for 3 clients + 1 global test set
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_tensor, y_tensor = torch.FloatTensor(X), torch.LongTensor(y)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Simulate Non-IID or distinct client datasets
    client_data = [
        (X_train[:200], y_train[:200]),
        (X_train[200:500], y_train[200:500]),
        (X_train[500:], y_train[500:])
    ]

    # Initialize and poorly train client models to simulate divergence
    client_models = [SimpleNN() for _ in range(3)]
    client_qualities = []

    for i, (Xc, yc) in enumerate(client_data):
        optimizer = optim.SGD(client_models[i].parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        # Train for just a few epochs to get distinct parameters
        for epoch in range(5):
            optimizer.zero_grad()
            out = client_models[i](Xc)
            loss = criterion(out, yc)
            loss.backward()
            optimizer.step()

        # Calculate local quality (Qi) to be used as fuzzy measure m({xi})
        acc = evaluate_model(client_models[i], Xc, yc)
        client_qualities.append(acc)

    print(f"Local Models Training Qualities (Q_i): {client_qualities}")

    # Dictionary to store final metrics
    experiment_metrics = {}

    # 1. FedAvg Experiment
    uniform_weights = [1 / 3, 1 / 3, 1 / 3]
    model_fedavg = aggregate_fedavg(SimpleNN(), client_models, uniform_weights)
    experiment_metrics['FedAvg (Mean)'] = evaluate_model(model_fedavg, X_test, y_test)

    # 2. Sugeno Integral Experiment
    model_sugeno = aggregate_integrals(SimpleNN(), client_models, client_qualities, method='sugeno')
    experiment_metrics['Sugeno Integral'] = evaluate_model(model_sugeno, X_test, y_test)

    # 3. Choquet Integral Experiment
    model_choquet = aggregate_integrals(SimpleNN(), client_models, client_qualities, method='choquet')
    experiment_metrics['Choquet Integral'] = evaluate_model(model_choquet, X_test, y_test)

    # 4. OWA Operator Experiment
    model_owa = aggregate_integrals(SimpleNN(), client_models, client_qualities, method='owa')
    experiment_metrics['OWA Operator'] = evaluate_model(model_owa, X_test, y_test)

    print("\nGlobal Evaluation Metrics:")
    for method, acc in experiment_metrics.items():
        print(f"{method}: {acc:.4f}")

    # Plot results
    # plot_metrics(experiment_metrics)