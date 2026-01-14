"""
Prediction model for bRate in the choices13k dataset.

This model predicts the frequency with which participants selected Gamble B
using features inspired by decision-making theories including:
- Expected Value (rational choice)
- Expected Utility Theory (risk aversion)
- Prospect Theory (loss aversion, probability weighting)
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the choices13k dataset."""
    selections = pd.read_csv('choices13k/c13k_selections.csv')
    with open('choices13k/c13k_problems.json', 'r') as f:
        problems = json.load(f)
    return selections, problems


def compute_expected_value(gamble):
    """Compute expected value of a gamble: sum(prob * outcome)."""
    return sum(prob * outcome for prob, outcome in gamble)


def compute_variance(gamble):
    """Compute variance of a gamble."""
    ev = compute_expected_value(gamble)
    return sum(prob * (outcome - ev) ** 2 for prob, outcome in gamble)


def prospect_value(x, alpha=0.88, lambda_loss=2.25):
    """
    Prospect Theory value function.
    - alpha: diminishing sensitivity parameter
    - lambda_loss: loss aversion coefficient
    """
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_loss * ((-x) ** alpha)


def probability_weight(p, gamma=0.61):
    """
    Prospect Theory probability weighting function (Prelec).
    Overweights small probabilities, underweights large ones.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return np.exp(-(-np.log(p)) ** gamma)


def compute_prospect_value(gamble, alpha=0.88, lambda_loss=2.25, gamma=0.61):
    """
    Compute the Prospect Theory value of a gamble.
    Uses cumulative prospect theory for multi-outcome gambles.
    """
    # Sort outcomes by value
    sorted_gamble = sorted(gamble, key=lambda x: x[1])

    # Separate gains and losses
    gains = [(p, o) for p, o in sorted_gamble if o >= 0]
    losses = [(p, o) for p, o in sorted_gamble if o < 0]

    pt_value = 0.0

    # Process gains (from highest to lowest)
    if gains:
        gains = sorted(gains, key=lambda x: x[1], reverse=True)
        cum_prob = 0.0
        for p, o in gains:
            new_cum = cum_prob + p
            weight = probability_weight(new_cum, gamma) - probability_weight(cum_prob, gamma)
            pt_value += weight * prospect_value(o, alpha, lambda_loss)
            cum_prob = new_cum

    # Process losses (from lowest to highest)
    if losses:
        losses = sorted(losses, key=lambda x: x[1])
        cum_prob = 0.0
        for p, o in losses:
            new_cum = cum_prob + p
            weight = probability_weight(new_cum, gamma) - probability_weight(cum_prob, gamma)
            pt_value += weight * prospect_value(o, alpha, lambda_loss)
            cum_prob = new_cum

    return pt_value


def compute_certainty_equivalent(gamble, risk_aversion=0.5):
    """Compute certainty equivalent using power utility."""
    ev = compute_expected_value(gamble)
    var = compute_variance(gamble)
    # Approximation: CE ≈ EV - 0.5 * risk_aversion * variance / EV (if EV > 0)
    if abs(ev) > 0.01:
        return ev - 0.5 * risk_aversion * var / abs(ev)
    return ev


def get_min_outcome(gamble):
    """Get minimum outcome of a gamble."""
    return min(o for _, o in gamble)


def get_max_outcome(gamble):
    """Get maximum outcome of a gamble."""
    return max(o for _, o in gamble)


def get_prob_loss(gamble):
    """Compute probability of a loss (negative outcome)."""
    return sum(p for p, o in gamble if o < 0)


def get_prob_gain(gamble):
    """Compute probability of a gain (positive outcome)."""
    return sum(p for p, o in gamble if o > 0)


def get_expected_loss(gamble):
    """Compute expected value of losses only."""
    losses = [(p, o) for p, o in gamble if o < 0]
    if not losses:
        return 0.0
    total_prob = sum(p for p, _ in losses)
    if total_prob == 0:
        return 0.0
    return sum(p * o for p, o in losses)


def get_expected_gain(gamble):
    """Compute expected value of gains only."""
    gains = [(p, o) for p, o in gamble if o > 0]
    if not gains:
        return 0.0
    return sum(p * o for p, o in gains)


def engineer_features(selections, problems):
    """
    Engineer features for predicting bRate.
    """
    features = []

    for idx in range(len(selections)):
        row = selections.iloc[idx]
        prob_data = problems[str(idx)]

        gamble_a = prob_data['A']
        gamble_b = prob_data['B']

        # Basic features from the dataset
        feat = {
            'Feedback': int(row['Feedback']),
            'Block': row['Block'],
            'Amb': int(row['Amb']),
            'Corr': row['Corr'],
            'LotShapeB': row['LotShapeB'],
            'LotNumB': row['LotNumB'],
        }

        # Expected Values
        ev_a = compute_expected_value(gamble_a)
        ev_b = compute_expected_value(gamble_b)
        feat['EV_A'] = ev_a
        feat['EV_B'] = ev_b
        feat['EV_diff'] = ev_b - ev_a  # Positive means B is better in EV
        feat['EV_ratio'] = ev_b / ev_a if abs(ev_a) > 0.01 else 0

        # Variance / Risk
        var_a = compute_variance(gamble_a)
        var_b = compute_variance(gamble_b)
        feat['Var_A'] = var_a
        feat['Var_B'] = var_b
        feat['Var_diff'] = var_b - var_a
        feat['Std_A'] = np.sqrt(var_a)
        feat['Std_B'] = np.sqrt(var_b)

        # Coefficient of variation (risk per unit return)
        feat['CV_A'] = np.sqrt(var_a) / abs(ev_a) if abs(ev_a) > 0.01 else 0
        feat['CV_B'] = np.sqrt(var_b) / abs(ev_b) if abs(ev_b) > 0.01 else 0

        # Prospect Theory values
        pt_a = compute_prospect_value(gamble_a)
        pt_b = compute_prospect_value(gamble_b)
        feat['PT_A'] = pt_a
        feat['PT_B'] = pt_b
        feat['PT_diff'] = pt_b - pt_a

        # Min/Max outcomes
        feat['Min_A'] = get_min_outcome(gamble_a)
        feat['Max_A'] = get_max_outcome(gamble_a)
        feat['Min_B'] = get_min_outcome(gamble_b)
        feat['Max_B'] = get_max_outcome(gamble_b)
        feat['Min_diff'] = feat['Min_B'] - feat['Min_A']
        feat['Max_diff'] = feat['Max_B'] - feat['Max_A']

        # Range (spread of outcomes)
        feat['Range_A'] = feat['Max_A'] - feat['Min_A']
        feat['Range_B'] = feat['Max_B'] - feat['Min_B']

        # Probability of loss/gain
        feat['ProbLoss_A'] = get_prob_loss(gamble_a)
        feat['ProbLoss_B'] = get_prob_loss(gamble_b)
        feat['ProbGain_A'] = get_prob_gain(gamble_a)
        feat['ProbGain_B'] = get_prob_gain(gamble_b)
        feat['ProbLoss_diff'] = feat['ProbLoss_B'] - feat['ProbLoss_A']

        # Expected gains and losses separately
        feat['ExpLoss_A'] = get_expected_loss(gamble_a)
        feat['ExpLoss_B'] = get_expected_loss(gamble_b)
        feat['ExpGain_A'] = get_expected_gain(gamble_a)
        feat['ExpGain_B'] = get_expected_gain(gamble_b)

        # Number of outcomes
        feat['NumOutcomes_A'] = len(gamble_a)
        feat['NumOutcomes_B'] = len(gamble_b)

        # Certainty equivalent
        feat['CE_A'] = compute_certainty_equivalent(gamble_a)
        feat['CE_B'] = compute_certainty_equivalent(gamble_b)
        feat['CE_diff'] = feat['CE_B'] - feat['CE_A']

        # Is gamble A certain? (variance = 0)
        feat['A_is_certain'] = int(var_a < 0.001)
        feat['B_is_certain'] = int(var_b < 0.001)

        # Skewness approximation (using third moment)
        ev_a_val = ev_a
        ev_b_val = ev_b
        skew_a = sum(p * ((o - ev_a_val) ** 3) for p, o in gamble_a)
        skew_b = sum(p * ((o - ev_b_val) ** 3) for p, o in gamble_b)
        feat['Skew_A'] = skew_a / (var_a ** 1.5) if var_a > 0.001 else 0
        feat['Skew_B'] = skew_b / (var_b ** 1.5) if var_b > 0.001 else 0

        # Dominance indicators
        feat['A_dominates_min'] = int(feat['Min_A'] >= feat['Min_B'])
        feat['A_dominates_max'] = int(feat['Max_A'] >= feat['Max_B'])
        feat['B_dominates_min'] = int(feat['Min_B'] >= feat['Min_A'])
        feat['B_dominates_max'] = int(feat['Max_B'] >= feat['Max_A'])

        # Safe vs risky classification
        feat['A_safer'] = int(var_a < var_b)
        feat['B_safer'] = int(var_b < var_a)

        # Interaction features
        feat['EV_diff_x_Feedback'] = feat['EV_diff'] * feat['Feedback']
        feat['PT_diff_x_Amb'] = feat['PT_diff'] * feat['Amb']

        features.append(feat)

    return pd.DataFrame(features)


def train_and_evaluate():
    """Train models and evaluate performance."""
    print("Loading data...")
    selections, problems = load_data()

    print("Engineering features...")
    X = engineer_features(selections, problems)
    y = selections['bRate'].values

    print(f"Dataset size: {len(X)} samples")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target (bRate) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")
    print()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models to try
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Create pipeline with scaling
        if name == 'Ridge':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([('model', model)])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        results[name] = {
            'model': pipeline,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test
        }

        print(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print()

    # Best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mse'])
    print(f"Best model: {best_model_name}")
    print(f"  Test MSE: {results[best_model_name]['test_mse']:.6f}")
    print(f"  Test RMSE: {np.sqrt(results[best_model_name]['test_mse']):.4f}")
    print(f"  Test MAE: {results[best_model_name]['test_mae']:.4f}")
    print(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")

    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        print(f"\nTop 15 Feature Importances ({best_model_name}):")
        model = results[best_model_name]['model'].named_steps['model']
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importances.head(15).to_string(index=False))

    return results, X_test, y_test


def predict_all():
    """Generate predictions for all data points."""
    print("Loading data...")
    selections, problems = load_data()

    print("Engineering features...")
    X = engineer_features(selections, problems)
    y = selections['bRate'].values

    print("Training final model on all data...")
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        min_samples_leaf=5, random_state=42
    )
    model.fit(X, y)

    # Generate predictions
    predictions = model.predict(X)

    # Clip to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    # Create output dataframe
    output = selections[['Problem', 'Feedback', 'bRate']].copy()
    output['predicted_bRate'] = predictions
    output['error'] = output['predicted_bRate'] - output['bRate']
    output['abs_error'] = np.abs(output['error'])

    # Save predictions
    output.to_csv('brate_predictions.csv', index=False)
    print(f"Predictions saved to brate_predictions.csv")

    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Mean Absolute Error: {output['abs_error'].mean():.4f}")
    print(f"  RMSE: {np.sqrt((output['error'] ** 2).mean()):.4f}")
    print(f"  R²: {r2_score(y, predictions):.4f}")
    print(f"  Correlation: {np.corrcoef(y, predictions)[0, 1]:.4f}")

    return output, model


if __name__ == '__main__':
    print("=" * 60)
    print("CHOICES13K BRATE PREDICTION MODEL")
    print("=" * 60)
    print()

    # Train and evaluate models
    results, X_test, y_test = train_and_evaluate()

    print()
    print("=" * 60)
    print("GENERATING FINAL PREDICTIONS")
    print("=" * 60)
    print()

    # Generate predictions for all data
    output, final_model = predict_all()
