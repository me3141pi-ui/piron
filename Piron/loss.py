import jax.numpy as np

# -------- Loss functions --------
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def binary_crossentropy(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def huber(y_pred, y_true, delta=1.0):
    err = y_pred - y_true
    is_small = np.abs(err) <= delta
    return np.mean(np.where(is_small, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta)))

def hinge(y_pred, y_true):
    # y_true ∈ {-1, +1}
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def squared_hinge(y_pred, y_true):
    return np.mean(np.maximum(0, 1 - y_true * y_pred) ** 2)

def kullback_leibler(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1)
    y_true = np.clip(y_true, eps, 1)
    return np.sum(y_true * np.log(y_true / y_pred)) / y_true.shape[0]

def cosine_proximity(y_pred, y_true, eps=1e-9):
    y_pred = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + eps)
    y_true = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + eps)
    return -np.mean(np.sum(y_pred * y_true, axis=1))

# -------- Derivatives --------
def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def mae_derivative(y_pred, y_true):
    return np.sign(y_pred - y_true) / y_true.size

def binary_crossentropy_derivative(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

def categorical_crossentropy_derivative(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true / (y_pred * y_true.shape[0])

def huber_derivative(y_pred, y_true, delta=1.0):
    err = y_pred - y_true
    is_small = np.abs(err) <= delta
    return np.where(is_small, err, delta * np.sign(err)) / y_true.size

def hinge_derivative(y_pred, y_true):
    grad = np.zeros_like(y_pred)
    mask = (1 - y_true * y_pred) > 0
    grad[mask] = -y_true[mask]
    return grad / y_true.size

def squared_hinge_derivative(y_pred, y_true):
    grad = np.zeros_like(y_pred)
    mask = (1 - y_true * y_pred) > 0
    grad[mask] = -2 * y_true[mask] * (1 - y_true[mask] * y_pred[mask])
    return grad / y_true.size

def kullback_leibler_derivative(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1)
    return -y_true / (y_pred * y_true.shape[0])

def cosine_proximity_derivative(y_pred, y_true, eps=1e-9):
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + eps)
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + eps)
    grad = -y_true_norm + (np.sum(y_pred_norm * y_true_norm, axis=1, keepdims=True) * y_pred_norm)
    return grad / y_true.shape[0]

# -------- Dictionaries --------
loss_dict = {
    "mse": mse,
    "mae": mae,
    "binary_crossentropy": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy,
    "huber": huber,
    "hinge": hinge,
    "squared_hinge": squared_hinge,
    "kl_divergence": kullback_leibler,
    "cosine_proximity": cosine_proximity,
}

loss_derivative_dict = {
    "mse": mse_derivative,
    "mae": mae_derivative,
    "binary_crossentropy": binary_crossentropy_derivative,
    "categorical_crossentropy": categorical_crossentropy_derivative,
    "huber": huber_derivative,
    "hinge": hinge_derivative,
    "squared_hinge": squared_hinge_derivative,
    "kl_divergence": kullback_leibler_derivative,
    "cosine_proximity": cosine_proximity_derivative,
}
