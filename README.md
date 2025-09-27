# Optimized Self-Developing Grey Model (OSDGM) in Python

This project is for **Self-Developing Grey Model (OSDGM)** with GM(1,1) and DGM models in Python for time series prediction. The code estimates model parameters, optimizes them and visualizes them.

## Features

- GM(1,1) prediction
- DGM prediction
- OSDGM prediction
- Mean Squared Error (MSE) loss 
- Export results to CSV and plot predictions

## Roadmap 
Takes a small time series as input:
```
x0 = np.array([21.1, 26.6, 36.1, 52.3, 80.1, 126.8])
```

Builds cumulative sums of the data to create a smooth series (x1). Grey models work on these accumulated sequences to better capture trends.

GM(1,1): Estimates a and b for the differential equation.

DGM: Estimates b1 and b2 for the discrete model.

OSDGM: Optimizes b3 using mean squared error (MSE) against true values, improving prediction accuracy.

Predicts future values.

Compares predictions with true values (if available) and outputs:

CSV file (osdgm_prediction_comparison.csv)

plot showing original, true, and predicted values.
