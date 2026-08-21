import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

def evaluate_model(y_pred,
                   y_test,
                   model_name="Model",
                   print_metrics = True,
                   plot=False
                   ):
    """
    Evaluate the predictions of a regression model.

    The function computes common regression metrics and can optionally
    visualize the relationship between actual and predicted values as well
    as the residual distribution.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted target values produced by the regression model.

    y_test : array-like of shape (n_samples,)
        True target values used to evaluate the predictions.

    model_name : str, default="Model"
        Name of the evaluated model. The name is used when printing the
        evaluation results.

    print_metrics : bool, default=True
        If True, print the calculated regression metrics.

    plot : bool, default=False
        If True, create three diagnostic plots:

        1. Predicted vs. actual values.
        2. Distributions of actual and predicted values.
        3. Distribution of residuals.

    Returns
    -------
    metrics : dict
        Dictionary containing the following regression metrics:

        - ``MAE`` : Mean Absolute Error.
        - ``RMSE`` : Root Mean Squared Error.
        - ``R2`` : Coefficient of determination (R²).

    Notes
    -----
    Residuals are calculated as::

        residual = y_test - y_pred

    Therefore:

    - Positive residuals indicate that the model underestimates the
      actual value.
    - Negative residuals indicate that the model overestimates the
      actual value.

    The residual plot automatically adjusts the x-axis so that both
    overestimation and underestimation remain visible even when the
    residual distribution is strongly asymmetric.

    Examples
    --------
    Evaluate predictions and print the regression metrics:

    >>> y_pred = model.predict(X_test)
    >>> metrics = evaluate_model(
    ...     y_pred,
    ...     y_test,
    ...     model_name="XGBoost",
    ... )

    Evaluate predictions and additionally display diagnostic plots:

    >>> metrics = evaluate_model(
    ...     y_pred,
    ...     y_test,
    ...     model_name="XGBoost",
    ...     plot=True,
    ... )
    """

    # Print model Name
    print(f"----- Evaluating {model_name} -----")

    # Ensure dimensions of predictions
    y_pred = np.array(y_pred).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    # Create Metric dict
    metrics = {
      "MAE": mean_absolute_error(y_test, y_pred),
      "RMSE": root_mean_squared_error(y_test, y_pred),
      "R2" : r2_score(y_test, y_pred),
    }


    # print metrics if needed
    if print_metrics:
      for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # create the plots if wanted
    if plot:
      fig, ax = plt.subplots(1, 3, figsize=(15, 5))

      # Plot 1: Scatter plot
      ax[0].scatter(y_test, y_pred, alpha = 0.3)
      ax[0].plot(y_test,y_test,color = "red")
      ax[0].set_xlabel("Actual values (y_test)")
      ax[0].set_ylabel("Predicted values (y_pred)")
      ax[0].set_title("Predicted vs Actual")

      # Plot 2: Histogram of true vs predicted
      sns.histplot(x=np.asarray(y_test).ravel(), ax=ax[1], label="Actual (y_test)", kde=True, color="red")
      sns.histplot(x=np.asarray(y_pred).ravel(), ax=ax[1], label="Predicted (y_pred)", kde=True, color= "blue")
      ax[1].set_xlabel("Value")
      ax[1].set_ylabel("Count")
      ax[1].set_title("Distribution of Actual vs Predicted")
      ax[1].legend()

      # Plot 3: Residuals
      residuals = y_test - y_pred

      sns.histplot(residuals, ax=ax[2], label="Residuals", kde=True)
      ax[2].axvline(x=0, color="red", linestyle="--")
      
      # Use adapted axis to show areas of over and underestimation
      x_min = min(min(residuals), -(max(residuals) / 2))
      x_max = max(max(residuals), -(min(residuals) / 2))
      ax[2].set_xlim(x_min, x_max)
      
      ax[2].set_xlabel("Residual")
      ax[2].set_ylabel("Count")
      ax[2].set_title("Residual Distribution")
      ax[2].text(0.8, 0.9, f"Model\nunderesitmates", transform = ax[2].transAxes, ha = "center")
      ax[2].text(0.15, 0.9, f"Model\noverestimates", transform = ax[2].transAxes, ha = "center")
      ax[2].legend()

      plt.tight_layout()
      plt.show()
      plt.close(fig)

    return metrics