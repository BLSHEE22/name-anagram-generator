import math
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Step 1: Feature computation for multiple starting words ---
def leftover_features_multiple(name, starting_words):
    """
    Computes aggregated features for a name with multiple N-letter starting words.
    Features:
        - mean_log_perm: average log permutations of leftovers
        - max_log_perm: max log permutations
        - var_log_perm: variance of log permutations
        - mean_entropy: average Shannon entropy of leftovers
        - var_entropy: variance of Shannon entropy
        - mean_leftover: average number of leftover letters
    """
    name = name.lower().replace(" ", "")
    name_counter = Counter(name)

    log_perm_list = []
    entropy_list = []
    leftover_counts = []

    for word in starting_words:
        start_counter = Counter(word.lower())
        leftover = name_counter - start_counter
        total_leftover = sum(leftover.values())
        leftover_counts.append(total_leftover)

        # log permutations
        log_perm = math.lgamma(total_leftover + 1)
        for c in leftover.values():
            log_perm -= math.lgamma(c + 1)
        log_perm_list.append(log_perm)

        # entropy
        entropy = 0
        for c in leftover.values():
            p = c / total_leftover
            entropy -= p * math.log(p)
        entropy_list.append(entropy)

    print(sum(leftover_counts))

    # Aggregate features
    features = [
        np.mean(log_perm_list),
        np.max(log_perm_list),
        np.var(log_perm_list),
        np.mean(entropy_list),
        np.var(entropy_list),
        np.mean(leftover_counts)
    ]
    return features

# --- Step 2: Prepare dataset ---
def prepare_dataset(names_dict, runtimes_dict):
    """
    Builds feature matrix X and target vector y from dictionaries:
        - names_dict: {name: list of starting words}
        - runtimes_dict: {name: measured runtime}
    """
    X = []
    y = []
    for name, words in names_dict.items():
        feats = leftover_features_multiple(name, words)
        X.append(feats)
        y.append(runtimes_dict[name])
    return np.array(X), np.array(y)

# --- Step 3: Fit regression model ---
def fit_runtime_model(X, y, log_transform=True):
    """
    Fits a linear regression model.
    log_transform: if True, regress on log(runtime) for exponential scaling.
    Returns the trained pipeline.
    """
    y_fit = np.log(y) if log_transform else y

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])

    pipeline.fit(X, y_fit)
    return pipeline

# --- Step 4: Predict runtime ---
def predict_runtime(model, names_dict, log_transform=True):
    """
    Predict runtime for new names given the trained model.
    """
    X_new = np.array([leftover_features_multiple(name, words) for name, words in names_dict.items()])
    y_pred = model.predict(X_new)
    return np.exp(y_pred) if log_transform else y_pred


#############################################################################################################

# uh... where's the data?

# Step 1: Prepare dataset
X, y = prepare_dataset(eight_letter_starters, search_durations)

# Step 2: Fit model
model = fit_runtime_model(X, y, log_transform=True)

# Step 3: Inspect coefficients
features = ["mean_log_perm", "max_log_perm", "var_log_perm", "mean_entropy", "var_entropy", "mean_leftover"]
print("Coefficients:")
for f, c in zip(features, model['reg'].coef_):
    print(f"{f}: {c:.4f}")
print("Intercept:", model['reg'].intercept_)

# Step 4: Predict runtime (for the same names or new names)
new_names = {"Jude Donabedian": ['undenied', 'unbodied', 'adjoined', 'enjoined', 'unjoined', 'aboideau', 'unbended', 'unbidden', 'unideaed', 'abounded']}


y_pred = predict_runtime(model, new_names)
print("Predicted runtimes:", y_pred)

