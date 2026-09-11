"""
Módulo Integral de Funciones de Agregación Avanzadas: 
Sugeno, Choquet, FG-Funcionales, Versiones Inspiradas, Generalizadas y su Extensión a Intervalos.
"""

import numpy as np

# ==========================================
# 1. FUNCIONES AUXILIARES Y DE INTERVALOS
# ==========================================

def xu_yager_key(interval):
    """
    Clave de ordenación para el orden admisible de Xu-Yager en L([0,1]).
    Ordena por el punto medio (inf + sup) y en caso de empate por la amplitud (sup - inf).
    """
    inf, sup = interval
    return (inf + sup, sup - inf)

def interval_min(A, B):
    """Mínimo reticular entre dos intervalos."""
    return [min(A[0], B[0]), min(A[1], B[1])]

def interval_max(A, B):
    """Máximo reticular entre dos intervalos."""
    return [max(A[0], B[0]), max(A[1], B[1])]

def interval_mul(A, B):
    """Multiplicación de intervalos (válido para elementos en [0, 1])."""
    return [max(0.0, min(1.0, A[0] * B[0])), max(0.0, min(1.0, A[1] * B[1]))]

def interval_sub(A, B):
    """Resta de intervalos asegurando contención en [0, 1]."""
    inf = max(0.0, A[0] - B[1])
    sup = max(inf, min(1.0, A[1] - B[0]))
    return [inf, sup]

def interval_scalar_mul(c, A):
    """Multiplicación de un escalar c en [0,1] por un intervalo A."""
    return [max(0.0, min(1.0, c * A[0])), max(0.0, min(1.0, c * A[1]))]


# ==========================================
# 2. FUNCIONES NUMÉRICAS (VALORES REALES)
# ==========================================

def sugeno_integral(x, m):
    """
    Integral de Sugeno clásica.
    x : vector de entrada (real)
    m : función de medida difusa que acepta una tupla/lista de índices de la coalición E_(i).
    """
    x = np.array(x, dtype=float)
    n = len(x)
    indices = np.argsort(x) # x_sigma(1) <= ... <= x_sigma(n)
    
    terms = []
    for i in range(n):
        # Coalición E_(sigma(i)) = elementos desde i hasta n en el orden creciente
        coalition = tuple(sorted(indices[i:]))
        term = min(x[indices[i]], m(coalition))
        terms.append(term)
    return max(terms)

def sugeno_fg_functional(x, m, F, G):
    """
    Sugeno-like FG-functional (Generalizada de Sugeno).
    F : función binaria (generaliza el mínimo)
    G : función n-aria (generaliza el máximo)
    """
    x = np.array(x, dtype=float)
    n = len(x)
    indices = np.argsort(x)
    
    terms = []
    for i in range(n):
        coalition = tuple(sorted(indices[i:]))
        term = F(x[indices[i]], m(coalition))
        terms.append(term)
    return G(*terms)

def sugeno_inspired_aggregation(x, H):
    """
    Función de agregación inspirada en Sugeno (SI_H clásica).
    H : función (n-1)-aria que evalúa la coalición restante x_hat_(i).
    """
    x_sorted = np.sort(np.array(x, dtype=float))
    n = len(x_sorted)
    
    terms = []
    for i in range(n):
        x_hat = np.concatenate([x_sorted[:i], x_sorted[i+1:]])
        term = min(x_sorted[i], H(x_hat))
        terms.append(term)
    return max(terms)

def generalized_sugeno_inspired_aggregation(x, F, G, H):
    """
    Función de agregación inspirada en Sugeno Generalizada.
    F : binaria (generaliza min), G : n-aria (generaliza max), H : (n-1)-aria (inspirada).
    """
    x_sorted = np.sort(np.array(x, dtype=float))
    n = len(x_sorted)
    
    terms = []
    for i in range(n):
        x_hat = np.concatenate([x_sorted[:i], x_sorted[i+1:]])
        term = F(x_sorted[i], H(x_hat))
        terms.append(term)
    return G(*terms)

def choquet_integral(x, m):
    """
    Integral de Choquet clásica.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    indices = np.argsort(x)
    
    x_sorted = x[indices]
    total = 0.0
    
    # Primer término: x_(1) * m(E_(1))
    coalition_1 = tuple(sorted(indices))
    total += x_sorted[0] * m(coalition_1)
    
    # Términos sucesivos: (x_(i) - x_(i-1)) * m(E_(i))
    for i in range(1, n):
        diff = x_sorted[i] - x_sorted[i-1]
        coalition_i = tuple(sorted(indices[i:]))
        total += diff * m(coalition_i)
        
    return total

def choquet_inspired_aggregation(x, H):
    """
    Función de agregación inspirada en Choquet.
    H : función que evalúa la coalición restante de elementos ordenados.
    """
    x_sorted = np.sort(np.array(x, dtype=float))
    n = len(x_sorted)
    
    total = x_sorted[0] * H(x_sorted)
    for i in range(1, n):
        diff = x_sorted[i] - x_sorted[i-1]
        coalition = x_sorted[i:]
        total += diff * H(coalition)
    return total


# ==========================================
# 3. EQUIVALENTES PARA INTERVALOS (INTERVAL-VALUED)
# ==========================================

def interval_sugeno_integral(intervals, m_iv):
    """
    Integral de Sugeno para datos en intervalos (IV-Sugeno).
    m_iv : medida difusa que mapea tuplas de índices a intervalos [inf, sup].
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    terms = []
    for i in range(n):
        # Para simular índices de coalición en intervalos ordenados por Xu-Yager:
        coalition = tuple(range(i, n))
        term = interval_min(intervals_sorted[i], m_iv(coalition))
        terms.append(term)
        
    # Máximo reticular de la lista de intervalos bajo orden Xu-Yager
    best = terms[0]
    for t in terms[1:]:
        if xu_yager_key(t) > xu_yager_key(best):
            best = t
    return best

def interval_sugeno_fg_functional(intervals, m_iv, F_iv, G_iv):
    """
    FG-functional de Sugeno para intervalos.
    F_iv : función binaria de intervalos, G_iv : función n-aria de intervalos.
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    terms = []
    for i in range(n):
        coalition = tuple(range(i, n))
        term = F_iv(intervals_sorted[i], m_iv(coalition))
        terms.append(term)
    return G_iv(terms)

def interval_sugeno_inspired_aggregation(intervals, H_iv):
    """
    Agregación inspirada en Sugeno para intervalos (SI_H para intervalos).
    H_iv : función que toma una lista de n-1 intervalos y devuelve un intervalo.
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    terms = []
    for i in range(n):
        x_hat = intervals_sorted[:i] + intervals_sorted[i+1:]
        term = interval_min(intervals_sorted[i], H_iv(x_hat))
        terms.append(term)
        
    best = terms[0]
    for t in terms[1:]:
        if xu_yager_key(t) > xu_yager_key(best):
            best = t
    return best

def generalized_interval_sugeno_inspired_aggregation(intervals, F_iv, G_iv, H_iv):
    """
    Agregación inspirada en Sugeno Generalizada para intervalos.
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    terms = []
    for i in range(n):
        x_hat = intervals_sorted[:i] + intervals_sorted[i+1:]
        term = F_iv(intervals_sorted[i], H_iv(x_hat))
        terms.append(term)
    return G_iv(terms)

def interval_choquet_integral(intervals, m_iv):
    """
    Integral de Choquet para intervalos (usando aritmética de intervalos).
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    # Primer término: X_(1) * m(E_(1))
    total = interval_mul(intervals_sorted[0], m_iv(tuple(range(0, n))))
    
    # Términos sucesivos: (X_(i) - X_(i-1)) * m(E_(i))
    for i in range(1, n):
        diff = interval_sub(intervals_sorted[i], intervals_sorted[i-1])
        prod = interval_mul(diff, m_iv(tuple(range(i, n))))
        
        # Suma de intervalos
        total = [max(0.0, min(1.0, total[0] + prod[0])), max(0.0, min(1.0, total[1] + prod[1]))]
        
    return total

def interval_choquet_inspired_aggregation(intervals, H_iv):
    """
    Agregación inspirada en Choquet para intervalos.
    """
    intervals_sorted = sorted([list(iv) for iv in intervals], key=xu_yager_key)
    n = len(intervals_sorted)
    
    total = interval_mul(intervals_sorted[0], H_iv(intervals_sorted))
    for i in range(1, n):
        diff = interval_sub(intervals_sorted[i], intervals_sorted[i-1])
        coalition = intervals_sorted[i:]
        prod = interval_mul(diff, H_iv(coalition))
        total = [max(0.0, min(1.0, total[0] + prod[0])), max(0.0, min(1.0, total[1] + prod[1]))]
    return total
