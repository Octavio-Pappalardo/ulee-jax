def compute_validation_metric(returns):
    validation_metric = returns[:, max(-10, -returns.shape[1]) :].mean()  # mean performance over the last 10 episodes
    return validation_metric
