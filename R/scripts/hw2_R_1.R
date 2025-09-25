################################################################################
# 1. COMMENTS
################################################################################

################################################################################
# 1.1 Suppose you estimate LASSO with a very large λ and with a very small λ
# Describe qualitatively how the coefficients and training/test error behave in both cases.
################################################################################

# 🔹 When λ is very large (the lasso is very short and tight):
# - The coefficients are heavily penalized, many shrink to zero.
# - The model becomes very simple → underfitting.
# - Training error: high, because the model does not fit well.
# - Test error: also high, since it fails to capture the true signal.
#
# 🔹 When λ is very small (the lasso is long and barely tight):
# - The coefficients are hardly penalized, they remain large.
# - The model fits the data too much → overfitting.
# - Training error: very low.
# - Test error: tends to be high because the model memorizes noise instead of generalizing.

################################################################################
# 1.2 Explain what cross-validation is and why it is useful in machine learning. 
# Illustrate with a sketch of how data is split.
################################################################################

# Cross-validation is a technique that splits the data into several groups called folds.
# In each iteration, one fold is used as validation and the others as training.
# This process is repeated until all folds have been used as validation.
# At the end, the results are averaged.
#
# This is useful in machine learning because:
# - It allows for a better evaluation of model performance.
# - It avoids relying on a single train/test split.
# - It helps in selecting hyperparameters (for example, the optimal value of λ in LASSO).

# Sketch (Figure 2): https://www.google.com/url?sa=i&url=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fcross_validation.html&psig=AOvVaw0nHZd8X_6_w8ONHLmEnLib&ust=1758774627929000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLj175nI8I8DFQAAAAAdAAAAABAE