################################################################################
# 2. LASSO
################################################################################

# Instalar y cargar el paquete readxl si aún no lo tienes
# install.packages("readxl")
library(readxl)

# Ruta al archivo Excel
file_path <- "Districtwise_literacy_rates.xlsx"

# Leer la primera hoja del Excel
df <- read_excel(path = file_path, sheet = 1)

# Ver las primeras filas
head(df)

################################################################################
# Keep only the observations with no missing values
################################################################################

# Elimina filas con al menos un NA
df_clean <- na.omit(df)

# Imprime número de filas antes y después
cat("Original rows:", nrow(df), "\n")
cat("Rows after dropping missing values:", nrow(df_clean), "\n")

################################################################################
# Create a histogram of the female and male literacy rate and comment briefly on its distribution.
################################################################################

# Instalar si no lo tienes
# install.packages("ggplot2")
library(ggplot2)

# Seleccionar las columnas
male_col <- "MALE_LIT"
female_col <- "FEMALE_LIT"

# Convertir a formato largo para graficar fácilmente
df_long <- data.frame(
  value = c(df_clean[[male_col]], df_clean[[female_col]]),
  group = rep(c("Male Literacy Rate", "Female Literacy Rate"),
              each = nrow(df_clean))
)

# Graficar histogramas lado a lado
p <- ggplot(df_long, aes(x = value, fill = group)) +
  geom_histogram(color = "black", bins = 20, alpha = 0.7) +
  facet_wrap(~ group, ncol = 2, scales = "free_y") +
  labs(x = "Literacy Rate (%)", y = "Number of Districts",
       title = "Distribution of Male and Female Literacy Rates (Census 2011)") +
  theme_minimal() +
  scale_fill_manual(values = c("blue", "red")) +
  theme(legend.position = "none")

print(p)

# --- Exportar (carpeta de trabajo actual) ---
# PNG (alta resolución)
ggsave(filename = "hist_literacy.png", plot = p, width = 10, height = 6, dpi = 300)

#Comment:
  
#Male Literacy Rate
#The histogram is concentrated in the higher range (≈75–95%).
#It looks roughly bell-shaped but skewed a bit left, meaning most districts have high male literacy, with fewer at the low end.
#This shows relatively less variation among districts.

#Female Literacy Rate
#The distribution is shifted to the left (≈50–80%), clearly lower than for males.
#There is a wider spread, with some districts below 40% female literacy, showing larger disparities.

# In short: male literacy is consistently higher and tightly clustered, while female literacy is more dispersed, highlighting the gender gap in literacy across districts in India.

################################################################################
# Estimate a low-dimensional specification and compute the (R^2) on the test set
################################################################################

### OLS ########################################################################

# Instalar si no lo tienes
# install.packages("caret")
library(caret)

set.seed(42)  # reproducibilidad

# Target
y <- df_clean$FEMALE_LIT

# Predictors (low-dimensional spec)
X <- df_clean[, c("TOTPOPULAT", "P_URB_POP", "SEXRATIO", "P_SC_POP", "P_ST_POP")]

# Train/test split (75% train, 25% test)
train_index <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Ajustar regresión lineal
model <- lm(y_train ~ ., data = data.frame(y_train, X_train))

# Predecir en test
y_pred <- predict(model, newdata = X_test)

# Calcular R^2 en test
r2 <- cor(y_test, y_pred)^2

cat("Low-dimensional specification:\n")
cat("Predictors:", paste(colnames(X), collapse = ", "), "\n")
cat(sprintf("R^2 on test set: %.4f\n", r2))

### LASSO ######################################################################

# install.packages("glmnet") # si hace falta
library(caret)
library(glmnet)

set.seed(42)  # reproducibilidad

# --- Target y predictores (low-dimensional) ---
y <- df_clean$FEMALE_LIT
X <- df_clean[, c("TOTPOPULAT", "P_URB_POP", "SEXRATIO", "P_SC_POP", "P_ST_POP")]

# --- Train/test split (75/25) ---
train_index <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[train_index, , drop = FALSE]
X_test  <- X[-train_index, , drop = FALSE]
y_train <- y[train_index]
y_test  <- y[-train_index]

# --- Matrices de diseño (sin intercepto; glmnet lo agrega) ---
mm_train <- model.matrix(~ ., data = as.data.frame(X_train))[, -1, drop = FALSE]
mm_test  <- model.matrix(~ ., data = as.data.frame(X_test))[,  -1, drop = FALSE]

# --- LASSO con validación cruzada (glmnet estandariza por defecto) ---
set.seed(42)
cvfit <- cv.glmnet(
  x = mm_train, y = y_train,
  alpha = 1,              # LASSO
  nfolds = 5,
  family = "gaussian",
  standardize = TRUE
)

# --- Predicciones en test ---
y_pred_min <- as.numeric(predict(cvfit, newx = mm_test, s = "lambda.min"))
y_pred_1se <- as.numeric(predict(cvfit, newx = mm_test, s = "lambda.1se"))

# --- R^2 en test ---
r2_min <- 1 - sum((y_test - y_pred_min)^2) / sum((y_test - mean(y_test))^2)
r2_1se <- 1 - sum((y_test - y_pred_1se)^2) / sum((y_test - mean(y_test))^2)

# --- # de variables seleccionadas (sin intercepto) ---
k_min <- sum(coef(cvfit, s = "lambda.min") != 0) - 1
k_1se <- sum(coef(cvfit, s = "lambda.1se") != 0) - 1

cat("Low-dimensional LASSO specification:\n")
cat("Predictors:", paste(colnames(X), collapse = ", "), "\n")
cat(sprintf("lambda.min = %.5g | active features = %d | R^2 test = %.4f\n", cvfit$lambda.min, k_min, r2_min))
cat(sprintf("lambda.1se = %.5g | active features = %d | R^2 test = %.4f\n", cvfit$lambda.1se, k_1se, r2_1se))

################################################################################
# Estimate a high-dimensional (flexible) specification: interaction terms and squared terms and compute the (R^2) on the test set
################################################################################

### Alternative 1 ##############################################################

# Paquetes (solo si quieres un split estilo sklearn)
# install.packages("caret")
# library(caret)

# set.seed(42)

# --- Target y predictores ---
#target <- "FEMALE_LIT"
#predictors <- c(
#  "TOTPOPULAT",  # población
#  "P_URB_POP",   # % urbana
#  "SEXRATIO",    # sex ratio
#  "P_SC_POP",    # % Scheduled Castes
#  "P_ST_POP",    # % Scheduled Tribes
#  "AREA_SQKM"    # área (km^2)
#)

# Data
#X <- df_clean[, predictors, drop = FALSE]
#y <- df_clean[[target]]

# --- Train/test split (75/25) ---
#idx_train <- createDataPartition(y, p = 0.75, list = FALSE)
#X_train <- X[idx_train, , drop = FALSE]
#X_test  <- X[-idx_train, , drop = FALSE]
#y_train <- y[idx_train]
#y_test  <- y[-idx_train]

# --- Construir fórmula: main effects + TODAS las interacciones de 2do orden + cuadrados ---
#main_str    <- paste(predictors, collapse = " + ")
#squares_str <- paste(sprintf("I(%s^2)", predictors), collapse = " + ")

# (main)^2 agrega main effects + TODAS las interacciones a 2do orden (parejas)
# añadimos además los términos cuadrados explícitos
#form_str <- paste0("~ (", main_str, ")^2 + ", squares_str)

# --- Expandir a matriz de diseño (sin columna de bias de poly, pero con intercepto del modelo) ---
# model.matrix genera automáticamente las dummies/expansiones según la fórmula
#mm_train <- model.matrix(as.formula(form_str), data = X_train)  # incluye Intercept
#mm_test  <- model.matrix(as.formula(form_str), data = X_test)

# Contar features expandidos "al estilo sklearn" (sin intercepto)
#n_in  <- ncol(X_train)
#n_out <- ncol(mm_train) - 1  # restamos intercepto

# --- Estandarizar columnas de features (excluye intercepto) ---
# Nota: OLS con y sin estandarización da mismas predicciones si no hay regularización,
# pero aquí replicamos tu pipeline.
#scale_train <- scale(mm_train[, -1, drop = FALSE])  # centra/escala con medias/SD del train
#center_vec  <- attr(scale_train, "scaled:center")
#scale_vec   <- attr(scale_train, "scaled:scale")

# aplicar el mismo centrado/escala al test
#scale_test <- sweep(mm_test[, -1, drop = FALSE], 2, center_vec, FUN = "-")
#scale_test <- sweep(scale_test, 2, scale_vec,   FUN = "/")

# reconstruimos matrices con intercepto (columna de 1s) + features escalados
#mm_train_scaled <- cbind(`(Intercept)` = 1, scale_train)
#mm_test_scaled  <- cbind(`(Intercept)` = 1, scale_test)

# --- Ajustar OLS en train y predecir en test ---
#fit <- lm.fit(x = mm_train_scaled, y = y_train)          # equivalente a lm(y ~ X)
#y_pred <- drop(mm_test_scaled %*% fit$coefficients)

# --- R^2 en test ---
#SSE <- sum((y_test - y_pred)^2)
#SST <- sum((y_test - mean(y_test))^2)
#r2  <- 1 - SSE/SST

#cat("High-dimensional (flexible) specification:\n")
#cat(sprintf("- Base predictors (%d): %s\n", n_in, paste(predictors, collapse = ", ")))
#cat(sprintf("- Expanded features (degree=2: squares + interactions): %d\n", n_out))
#cat(sprintf("- Test R^2: %.4f\n", r2))

### Alternative 2 ##############################################################

# install.packages("caret")  # si no lo tienes
#library(caret)

#set.seed(42)

# ===== 1) BASE VARIABLES =====
#base_cols <- c(
#  "FEMALE_LIT",  # target
#  "TOTPOPULAT",  # población (miles)
#  "P_URB_POP",   # % urbano
#  "SEXRATIO",    # sex ratio
#  "P_SC_POP",    # % Scheduled Castes
#  "P_ST_POP",    # % Scheduled Tribes
#  "AREA_SQKM"    # área (km^2)
#)

#dfm <- df_clean

# ===== 2) EDUCATION INFRASTRUCTURE (aggregate por prefijo) =====
#sum_prefix <- function(d, prefix, start, end) {
#  cols <- paste0(prefix, start:end)
#  cols <- cols[cols %in% names(d)]
#  if (length(cols) == 0) return(rep(NA_real_, nrow(d)))
#  rowSums(d[, cols, drop = FALSE], na.rm = TRUE)
#}

#dfm$SCHTOT   <- if ("SCHTOT" %in% names(dfm)) dfm$SCHTOT else sum_prefix(dfm, "SCH",   1, 9)
#dfm$TCHTOT   <- sum_prefix(dfm, "TCH",   1, 7)
#dfm$CLSTOT   <- sum_prefix(dfm, "CLS",   1, 7)
#dfm$SELETOT  <- sum_prefix(dfm, "SELE",  1, 7)
#dfm$SCOMPTOT <- sum_prefix(dfm, "SCOMP", 1, 7)
#dfm$ENR50TOT <- sum_prefix(dfm, "ENR50", 1, 9)

# ===== 3) DERIVED FEATURES =====
#pop_persons <- dfm$TOTPOPULAT * 1000

#safe_div <- function(a, b) {
#  out <- as.numeric(a) / as.numeric(b)
#  out[is.infinite(out)] <- NA_real_
#  out
#}

#dfm$pop_density           <- safe_div(pop_persons, dfm$AREA_SQKM)
#dfm$schools_per_100k      <- safe_div(dfm$SCHTOT, (pop_persons / 100000))
#dfm$schools_per_100sqkm   <- safe_div(dfm$SCHTOT, (dfm$AREA_SQKM / 100))
#dfm$teachers_per_school   <- safe_div(dfm$TCHTOT, dfm$SCHTOT)
#dfm$classrooms_per_school <- safe_div(dfm$CLSTOT, dfm$SCHTOT)
#dfm$elec_share            <- safe_div(dfm$SELETOT, dfm$SCHTOT)
#dfm$comp_share            <- safe_div(dfm$SCOMPTOT, dfm$SCHTOT)
#dfm$small_enrol_share     <- safe_div(dfm$ENR50TOT, dfm$SCHTOT)

# ===== 4) SQUARED TERMS =====
#sq_cols <- c("P_URB_POP","schools_per_100k","teachers_per_school",
#             "elec_share","comp_share","pop_density")
#for (col in sq_cols) {
#  if (col %in% names(dfm)) {
#    dfm[[paste0(col, "__sq")]] <- dfm[[col]]^2
#  }
#}

# ===== 5) INTERACTIONS =====
#inter <- function(d, a, b) {
#  nm <- paste0(a, "__x__", b)
#  d[[nm]] <- d[[a]] * d[[b]]
#  list(data = d, name = nm)
#}

#interaction_terms <- character(0)
#add_inter <- function(a, b) {
#  if (a %in% names(dfm) && b %in% names(dfm)) {
#    res <- inter(dfm, a, b)
#    assign("dfm", res$data, inherits = TRUE)
#    assign("interaction_terms", c(get("interaction_terms", inherits = TRUE), res$name), inherits = TRUE)
#  }
#}

#add_inter("P_URB_POP","elec_share")
#add_inter("P_URB_POP","comp_share")
#add_inter("P_URB_POP","P_SC_POP")
#add_inter("P_URB_POP","SEXRATIO")
#add_inter("P_ST_POP","AREA_SQKM")
#add_inter("schools_per_100k","teachers_per_school")
#add_inter("teachers_per_school","comp_share")
#add_inter("small_enrol_share","AREA_SQKM")
#add_inter("pop_density","schools_per_100sqkm")

# ===== 6) Predictores finales =====
#predictors <- c(
#  "TOTPOPULAT","P_URB_POP","SEXRATIO","P_SC_POP","P_ST_POP","AREA_SQKM",
#  "SCHTOT","TCHTOT","CLSTOT",
#  "pop_density","schools_per_100k","schools_per_100sqkm",
#  "teachers_per_school","classrooms_per_school",
#  "elec_share","comp_share","small_enrol_share",
#  paste0(sq_cols, "__sq")[paste0(sq_cols, "__sq") %in% names(dfm)],
#  interaction_terms
#)

#needed_cols <- unique(c("FEMALE_LIT", predictors))
#dfm_model <- dfm[, needed_cols, drop = FALSE]

# Reemplazar Inf por NA y dropear filas con NA
#for (j in seq_along(dfm_model)) {
#  v <- dfm_model[[j]]
#  v[is.infinite(v)] <- NA_real_
#  dfm_model[[j]] <- v
#}
#dfm_model <- na.omit(dfm_model)

# ===== 7) Split, scale y OLS =====
#X <- dfm_model[, predictors, drop = FALSE]
#y <- as.numeric(dfm_model$FEMALE_LIT)

#idx_train <- createDataPartition(y, p = 0.75, list = FALSE)
#X_train <- X[idx_train, , drop = FALSE]
#X_test  <- X[-idx_train, , drop = FALSE]
#y_train <- y[idx_train]
#y_test  <- y[-idx_train]

# Estandarizar (solo features)
#scale_train <- scale(X_train)  # centra y escala con medias/SD del train
#center_vec  <- attr(scale_train, "scaled:center")
#scale_vec   <- attr(scale_train, "scaled:scale")

#scale_test <- sweep(X_test, 2, center_vec, FUN = "-")
#scale_test <- sweep(scale_test, 2, scale_vec,   FUN = "/")

# Ajustar OLS sobre features escalados
#train_df <- data.frame(y_train, scale_train)
#fit <- lm(y_train ~ ., data = train_df)

# Predicción en test y R^2
#y_pred <- predict(fit, newdata = data.frame(scale_test))
#SSE <- sum((y_test - y_pred)^2)
#SST <- sum((y_test - mean(y_test))^2)
#r2  <- 1 - SSE/SST

#cat("Smarter high-dimensional model\n")
#cat(sprintf("- n_features: %d\n", ncol(X_train)))
#cat(sprintf("- Test R^2: %.4f\n", r2))

### Alternative 3  #############################################################

### OLS ########################################################################

# install.packages("caret")  # si no lo tienes
library(caret)

set.seed(42)

TARGET <- "FEMALE_LIT"

# 1) Variables base (demografía + educación)
demographic <- c(
  "TOTPOPULAT","P_URB_POP","POPULATION_0_6","GROWTHRATE","SEXRATIO",
  "P_SC_POP","P_ST_POP","AREA_SQKM","TOT_6_10_15","TOT_11_13_15"
)

sch_gov      <- c("SCH2G","SCH3G","SCH4G","SCH5G","SCH6G","SCH7G")
sch_priv     <- c("SCH2P","SCH3P","SCH4P","SCH5P","SCH6P","SCH7P")
sch_gov_rur  <- c("SCH2GR","SCH3GR","SCH4GR","SCH5GR","SCH6GR","SCH7GR")
sch_priv_rur <- c("SCH2PR","SCH3PR","SCH4PR","SCH5PR","SCH6PR","SCH7PR")

sele  <- c("SELE2","SELE3","SELE4","SELE5","SELE6","SELE7")
scomp <- c("SCOMP2","SCOMP3","SCOMP4","SCOMP5","SCOMP6","SCOMP7")
enr50 <- c("ENR502","ENR503","ENR504","ENR505","ENR506","ENR507")
cls   <- c("CLS2","CLS3","CLS4","CLS5","CLS6","CLS7")
tch   <- c("TCH2","TCH3","TCH4","TCH5","TCH6","TCH7")

base_predictors <- c(
  demographic,
  sch_gov, sch_priv, sch_gov_rur, sch_priv_rur,
  sele, scomp, enr50, cls, tch
)

existing <- intersect(base_predictors, names(df_clean))
missing  <- setdiff(base_predictors, names(df_clean))
if (length(missing) > 0) {
  cat("[warning] Missing variables skipped:", paste(missing, collapse = ", "), "\n")
}

dfm <- df_clean[, c(TARGET, existing), drop = FALSE]

# 2) Términos no lineales (cuadrados)
squares <- c("P_URB_POP","SEXRATIO","P_SC_POP","P_ST_POP","TOTPOPULAT","AREA_SQKM")
for (col in squares) {
  if (col %in% names(dfm)) {
    dfm[[paste0(col, "__sq")]] <- dfm[[col]]^2
  }
}

# 3) Interacciones clave
interactions <- list(
  c("P_URB_POP", "SEXRATIO"),
  c("P_URB_POP", "P_SC_POP"),
  c("P_URB_POP", "P_ST_POP"),
  c("SELE2", "SCOMP2"),
  c("CLS2",  "TCH2")
)
for (pair in interactions) {
  a <- pair[1]; b <- pair[2]
  if (a %in% names(dfm) && b %in% names(dfm)) {
    dfm[[paste0(a, "__x__", b)]] <- dfm[[a]] * dfm[[b]]
  }
}

# 4) Definir X, y
predictors <- setdiff(names(dfm), TARGET)
X <- dfm[, predictors, drop = FALSE]
y <- as.numeric(dfm[[TARGET]])

# Limpiar: reemplazar Inf por NA y omitir filas con NA
for (j in seq_along(X)) {
  v <- as.numeric(X[[j]])
  v[is.infinite(v)] <- NA_real_
  X[[j]] <- v
}
keep <- complete.cases(X, y)
X <- X[keep, , drop = FALSE]
y <- y[keep]

# 5) Split train/test (75/25)
idx_train <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[idx_train, , drop = FALSE]
X_test  <- X[-idx_train, , drop = FALSE]
y_train <- y[idx_train]
y_test  <- y[-idx_train]

# 6) Standardize (solo features): como StandardScaler de sklearn
Xtr_sc <- scale(X_train)
ctr <- attr(Xtr_sc, "scaled:center")
scl <- attr(Xtr_sc, "scaled:scale")

Xte_sc <- sweep(X_test, 2, ctr, FUN = "-")
Xte_sc <- sweep(Xte_sc, 2, scl, FUN = "/")

# 7) OLS sobre features escalados
train_df <- data.frame(y_train, Xtr_sc, check.names = FALSE)
fit <- lm(y_train ~ ., data = train_df)

# Predicción en test y R^2
y_pred <- predict(fit, newdata = data.frame(Xte_sc, check.names = FALSE))
SSE <- sum((y_test - y_pred)^2)
SST <- sum((y_test - mean(y_test))^2)
r2  <- 1 - SSE/SST

cat("\n=== Wide model with squares + selected interactions ===\n")
cat(sprintf("n_features: %d\n", ncol(X_train)))
cat(sprintf("Test R^2: %.4f\n", r2))

### LASSO ######################################################################

# install.packages("glmnet")  # si hace falta
library(glmnet)

# Convierte a matrices numéricas (glmnet requiere matrix)
Xtr_mat <- as.matrix(X_train)  # usa X_train sin escalar; glmnet escalará internamente
Xte_mat <- as.matrix(X_test)

set.seed(42)
cvfit <- cv.glmnet(
  x = Xtr_mat, y = y_train,
  alpha = 1,            # LASSO
  nfolds = 5,
  family = "gaussian",
  standardize = TRUE    # estandariza con medias/SD del TRAIN
)

# Predicciones en test
yhat_min <- as.numeric(predict(cvfit, newx = Xte_mat, s = "lambda.min"))
yhat_1se <- as.numeric(predict(cvfit, newx = Xte_mat, s = "lambda.1se"))

# Métricas en test
r2_min <- 1 - sum((y_test - yhat_min)^2) / sum((y_test - mean(y_test))^2)
r2_1se <- 1 - sum((y_test - yhat_1se)^2) / sum((y_test - mean(y_test))^2)

# Nº de features activas (sin intercepto)
k_min <- sum(coef(cvfit, s = "lambda.min") != 0) - 1
k_1se <- sum(coef(cvfit, s = "lambda.1se") != 0) - 1

cat("\n=== Wide LASSO: squares + selected interactions ===\n")
cat(sprintf("Input features: %d\n", ncol(X_train)))
cat(sprintf("lambda.min = %.5g | active = %d | Test R^2 = %.4f\n",
            cvfit$lambda.min, k_min, r2_min))
cat(sprintf("lambda.1se = %.5g | active = %d | Test R^2 = %.4f\n",
            cvfit$lambda.1se, k_1se, r2_1se))

################################################################################
# For λ ranging from 10,000 down to 0.001, plot the path of the number of nonzero coefficients and briefly comment on the result.
################################################################################

# install.packages("glmnet")  # si no lo tienes
library(glmnet)

# 6.1 Estandarizar con medias/SD del TRAIN
Xtr <- as.matrix(X_train)
Xte <- as.matrix(X_test)
ytr <- as.numeric(y_train)

Xtr_sc <- scale(Xtr)                      # centra/escala con stats del train
center_vec <- attr(Xtr_sc, "scaled:center")
scale_vec  <- attr(Xtr_sc, "scaled:scale")

Xte_sc <- sweep(Xte, 2, center_vec, FUN = "-")
Xte_sc <- sweep(Xte_sc, 2, scale_vec,  FUN = "/")

# 6.2 Grid de lambdas: 10,000 → 0.001 (escala log)
lambdas <- exp(seq(log(1e4), log(1e-3), length.out = 100))

# Ajuste único de LASSO sobre todo el grid
# (alpha=1 LASSO; standardize=FALSE porque ya escalamos nosotros)
fit <- glmnet(
  x = Xtr_sc, y = ytr,
  alpha = 1,
  lambda = lambdas,
  standardize = FALSE,
  intercept = TRUE,
  family = "gaussian"
)

# Nº de coeficientes distintos de 0 para cada lambda (excluye intercepto)
# fit$beta es una matriz p x |lambda|
nonzeros <- colSums(as.matrix(fit$beta != 0))

# 6.3 Graficar
plot(
  x = lambdas, y = nonzeros, type = "o",
  log = "x", xlab = "Lambda (log scale)", ylab = "Number of nonzero coefficients",
  main = "LASSO Path — Active Coefficients vs Lambda (Wide model)"
)
grid()

# Imprimir resumen como en Python
cat(sprintf("Total candidate features: %d\n", ncol(X_train)))
cat(sprintf("Nonzeros at λ=%.0f: %d\n", lambdas[1], nonzeros[1]))
cat(sprintf("Nonzeros at λ=%.3f: %d\n", lambdas[length(lambdas)], nonzeros[length(nonzeros)]))

#Comment:
  
#At very large λ (10,000), the penalty is so strong that all coefficients shrink to zero → no predictors are selected.
#At small λ (<10⁻¹), nearly all 75 candidate features enter the model.
#The path shows that LASSO performs automatic variable selection: only the most important predictors remain nonzero at moderate λ, while less relevant ones are eliminated.
