################################################################################
# 3. Potential Outcomes and RCTs
################################################################################

################################################################################
# 3.1 Data Simulation 
################################################################################

################################################################################

set.seed(123)   # para reproducibilidad
n <- 1000       # número de observaciones

# Covariables
x1 <- rnorm(n, mean = 0, sd = 1)          # normal(0,1)
x2 <- rnorm(n, mean = 0, sd = 1)          # normal(0,1)
x3 <- rbinom(n, size = 1, prob = 0.5)  # Bernoulli(0.5)
x4 <- rnorm(n, mean = 0, sd = 1)          # normal(0,1)

# Tratamiento D ~ Bernoulli(0.5)
D <- rbinom(n, size = 1, prob = 0.5)

# Error
epsilon <- rnorm(n, mean = 0, sd = 1)

# Variable de resultado
Y <- 2*D + 0.5*x1 - 0.3*x2 + 0.2*x3 + epsilon

# Crear DataFrame
df <- data.frame(
  Y = Y, D = D, x1 = x1, x2 = x2, x3 = x3, x4 = x4
)

# Primeras filas del dataset simulado
cat("Primeras filas del dataset simulado:\n")
head(df)

################################################################################

cat("\n=== Balance de covariables entre Tratamiento y Control ===\n")

for (var in c("x1", "x2", "x3", "x4")) {
  mean_treat   <- mean(df[df$D == 1, var])
  mean_control <- mean(df[df$D == 0, var])
  ttest <- t.test(df[df$D == 1, var], df[df$D == 0, var])
  
  cat(sprintf("%s: media tratados=%.3f, media control=%.3f, p-valor=%.3f\n",
              var, mean_treat, mean_control, ttest$p.value))
}

################################################################################
# Estimating the Average Treatment Effect (3 points)
################################################################################

################################################################################

cat("\n=== Estimación del ATE con regresión simple (Y ~ D) ===\n")
model1 <- lm(Y ~ D, data = df)
summary(model1)$coefficients  # tabla de coeficientes

################################################################################

cat("\n=== Estimación del ATE con controles (Y ~ D + x1 + x2 + x3 + x4) ===\n")
model2 <- lm(Y ~ D + x1 + x2 + x3 + x4, data = df)
summary(model2)$coefficients

################################################################################

# Comparación de coeficientes
coef_simple   <- coef(summary(model1))["D", "Estimate"]
se_simple     <- coef(summary(model1))["D", "Std. Error"]

coef_control  <- coef(summary(model2))["D", "Estimate"]
se_control    <- coef(summary(model2))["D", "Std. Error"]

cat("\n=== Comparación de ATE ===\n")
cat(sprintf("ATE (sin controles): %.3f (SE=%.3f)\n", coef_simple, se_simple))
cat(sprintf("ATE (con controles): %.3f (SE=%.3f)\n", coef_control, se_control))

## ¿El ATE cambia? ¿Qué pasa con los errores estándar?
## El ATE prácticamente no cambia al incluir controles, lo que confirma que la asignación de tratamiento fue aleatoria. Sin embargo, los errores estándar se reducen cuando controlamos covariables, lo que implica una ganancia en eficiencia estadística.

################################################################################
# 3.3 LASSO and Variable Selection
################################################################################

################################################################################

# install.packages("glmnet") # si no lo tienes
suppressPackageStartupMessages(library(glmnet))

lasso_then_ate <- function(df, y = "Y", d = "D", xs = c("x1","x2","x3","x4"),
                           nfolds = 5, standardize = TRUE, seed = 123) {
  stopifnot(is.data.frame(df))
  
  # --- Verificaciones de columnas
  need <- c(y, d, xs)
  miss <- setdiff(need, names(df))
  if (length(miss)) stop("Faltan columnas: ", paste(miss, collapse = ", "))
  
  # --- Subset y coersiones mínimas
  df0 <- df[, need, drop = FALSE]
  if (!is.numeric(df0[[y]])) df0[[y]] <- suppressWarnings(as.numeric(df0[[y]]))
  for (v in xs) if (!is.numeric(df0[[v]])) df0[[v]] <- suppressWarnings(as.numeric(df0[[v]]))
  if (is.logical(df0[[d]])) df0[[d]] <- as.integer(df0[[d]])
  
  ok  <- complete.cases(df0)
  dfc <- df0[ok, , drop = FALSE]
  if (!nrow(dfc)) stop("No hay filas completas tras eliminar NA.")
  
  # --- Matriz X (sin D) e y
  X  <- as.matrix(dfc[, xs, drop = FALSE])
  yv <- as.numeric(dfc[[y]])
  
  # Remover varianza cero
  nzv <- apply(X, 2, function(z) var(z, na.rm = TRUE) > 0)
  if (!all(nzv)) {
    X  <- X[, nzv, drop = FALSE]
    xs <- xs[nzv]
  }
  if (!ncol(X)) stop("No quedan predictores después de remover varianza cero.")
  
  # --- LASSO con CV
  set.seed(seed)
  cvfit <- cv.glmnet(x = X, y = yv, alpha = 1, nfolds = nfolds,
                     standardize = standardize, family = "gaussian")
  lambda_min <- cvfit$lambda.min
  
  # --- Seleccionadas en λ_min
  coef_min <- as.matrix(coef(cvfit, s = "lambda.min"))
  sel_vars <- setdiff(rownames(coef_min)[coef_min[,1] != 0], "(Intercept)")
  
  # --- Re-estimar ATE con solo D + X_selected
  rhs <- unique(c(d, sel_vars))
  form <- as.formula(paste(y, "~", paste(rhs, collapse = " + ")))
  ols  <- lm(form, data = dfc)
  ct   <- summary(ols)$coefficients
  
  # Extraer ATE (coef de D)
  ate   <- unname(ct["D", "Estimate"])
  se    <- unname(ct["D", "Std. Error"])
  pval  <- unname(ct["D", "Pr(>|t|)"])
  ci    <- confint(ols, "D", level = 0.95)
  
  list(
    lambda_min    = lambda_min,
    selected_vars = setdiff(sel_vars, d),  # por si acaso
    formula_ols   = form,
    ols           = ols,
    coeftable     = ct,
    ate           = ate,
    se            = se,
    pval          = pval,
    ci95          = as.numeric(ci)
  )
}

# ================== EJECUCIÓN ==================
res <- lasso_then_ate(df = df, y = "Y", d = "D", xs = c("x1","x2","x3","x4"))

cat("lambda.min:", res$lambda_min, "\n")
cat("Selected at lambda.min:",
    if (length(res$selected_vars)) paste(res$selected_vars, collapse = ", ") else "(none)","\n")
cat("OLS formula:", deparse(res$formula_ols), "\n")

cat(sprintf("ATE (coef D) = %.3f, SE = %.3f, p = %.4f, 95%% CI = [%.3f, %.3f]\n",
            res$ate, res$se, res$pval, res$ci95[1], res$ci95[2]))

################################################################################

# Comment

#Part B – no controls: ATE = 2.107, SE = 0.072
#Part B – with controls: ATE = 2.056, SE = 0.063
#Post-LASSO (D + X_selected): ATE = 2.056, SE = 0.063, 95% CI = [1.932, 2.180]

#In this case, with the specific variables we have used, post-LASSO matches OLS with controls—with only 4 predictors and clear signal, LASSO doesn’t need to drop variables at λ_min.
#Why use Lasso?
# in higher-dimensional settings (many X’s, interactions, polynomials), LASSO keeps prognostic covariates and drops noise.
# Better generalization & stability: mitigates overfitting and collinearity; can yield smaller variance for the ATE after re-estimating Y - D + Selected Variables
