import pytest
import numpy as np

# Importamos las funciones del módulo principal (asumiendo que se llama aggregation_suite.py)
from fuzzy_integrals import (
    sugeno_integral,
    sugeno_fg_functional,
    sugeno_inspired_aggregation,
    generalized_sugeno_inspired_aggregation,
    choquet_integral,
    choquet_inspired_aggregation,
    interval_sugeno_integral,
    interval_sugeno_fg_functional,
    interval_sugeno_inspired_aggregation,
    generalized_interval_sugeno_inspired_aggregation,
    interval_choquet_integral,
    interval_choquet_inspired_aggregation,
)

# ==========================================
# FIXTURES (Datos y funciones de soporte)
# ==========================================

@pytest.fixture
def numeric_inputs():
    return [0.2, 0.8, 0.5]

@pytest.fixture
def dummy_measure():
    return lambda coalition: len(coalition) / 3

@pytest.fixture
def h_func():
    return lambda x_hat: np.mean(x_hat) if len(x_hat) > 0 else 0.0

@pytest.fixture
def interval_inputs():
    return [[0.1, 0.3], [0.7, 0.9], [0.4, 0.6]]

@pytest.fixture
def dummy_measure_iv():
    return lambda coalition: [len(coalition) / 3, min(1.0, len(coalition) / 3 + 0.1)]

@pytest.fixture
def h_iv_func():
    return lambda x_hat: [np.mean([iv[0] for iv in x_hat]), np.mean([iv[1] for iv in x_hat])] if len(x_hat) > 0 else [0, 0]


# ==========================================
# 1. TESTS PARA FUNCIONES NUMÉRICAS (REALES)
# ==========================================

def test_sugeno_integral(numeric_inputs, dummy_measure):
    res = sugeno_integral(numeric_inputs, dummy_measure)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_sugeno_fg_functional(numeric_inputs, dummy_measure):
    fg_f = lambda a, b: a * b
    fg_g = lambda *args: min(1.0, sum(args))
    res = sugeno_fg_functional(numeric_inputs, dummy_measure, fg_f, fg_g)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_sugeno_inspired_aggregation(numeric_inputs, h_func):
    res = sugeno_inspired_aggregation(numeric_inputs, h_func)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_generalized_sugeno_inspired_aggregation(numeric_inputs, h_func):
    res = generalized_sugeno_inspired_aggregation(
        numeric_inputs, 
        F=lambda a, b: a * b, 
        G=lambda *args: sum(args), 
        H=h_func
    )
    assert isinstance(res, float)

def test_choquet_integral(numeric_inputs, dummy_measure):
    res = choquet_integral(numeric_inputs, dummy_measure)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_choquet_inspired_aggregation(numeric_inputs):
    h_coalition = lambda coalition: np.mean(coalition)
    res = choquet_inspired_aggregation(numeric_inputs, h_coalition)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0


# ==========================================
# 2. TESTS PARA FUNCIONES DE INTERVALOS (IV)
# ==========================================

def test_interval_sugeno_integral(interval_inputs, dummy_measure_iv):
    res = interval_sugeno_integral(interval_inputs, dummy_measure_iv)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]  # Límite inferior menor o igual al superior
    assert 0.0 <= res[0] <= res[1] <= 1.0

def test_interval_sugeno_fg_functional(interval_inputs, dummy_measure_iv):
    f_iv = lambda a, b: [a[0] * b[0], a[1] * b[1]]
    g_iv = lambda terms: [min([t[0] for t in terms]), max([t[1] for t in terms])]
    res = interval_sugeno_fg_functional(interval_inputs, dummy_measure_iv, f_iv, g_iv)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]

def test_interval_sugeno_inspired_aggregation(interval_inputs, h_iv_func):
    res = interval_sugeno_inspired_aggregation(interval_inputs, h_iv_func)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]

def test_generalized_interval_sugeno_inspired_aggregation(interval_inputs, h_iv_func):
    f_iv = lambda a, b: [a[0] * b[0], a[1] * b[1]]
    g_iv = lambda terms: [min([t[0] for t in terms]), max([t[1] for t in terms])]
    res = generalized_interval_sugeno_inspired_aggregation(interval_inputs, f_iv, g_iv, h_iv_func)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]

def test_interval_choquet_integral(interval_inputs, dummy_measure_iv):
    res = interval_choquet_integral(interval_inputs, dummy_measure_iv)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]

def test_interval_choquet_inspired_aggregation(interval_inputs):
    h_iv_coalition = lambda coalition: [np.mean([iv[0] for iv in coalition]), np.mean([iv[1] for iv in coalition])]
    res = interval_choquet_inspired_aggregation(interval_inputs, h_iv_coalition)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] <= res[1]
