# Chapter 6: Explainable AI

## Why XAI?

A model that says "Positive 75%" isn't enough. We need to know **why**.

## Methods Used

### SHAP (SHapley Additive exPlanations)

Shows word-level importance based on game theory.

**Optimized with**: Genetic Algorithm
- n_samples: 150
- max_evals: 350

### LIME (Local Interpretable Model-agnostic Explanations)

Creates local linear approximations around predictions.

**Optimized with**: Harmony Search
- kernel_width: 1.2
- num_features: 12

### Grad-CAM

Visualizes attention patterns in the model.

**Optimized with**: Firefly Algorithm
- layer_index: -2
- threshold: 0.45

## Example Output

For review: "This movie was absolutely fantastic!"

| Word | SHAP Value | Contribution |
|------|------------|--------------|
| fantastic | +0.35 | Positive |
| absolutely | +0.12 | Positive |
| movie | +0.02 | Neutral |

## Conclusion

XAI makes our model **trustworthy** and **debuggable**.
