y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)

log_reg = make_pipeline(
    StandardScaler(),
    SelectPercentile(percentile=50),
    LogisticRegression(solver="liblinear", penalty="l1")
)
log_reg.set_output(transform="pandas")

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)

feature_names = log_reg[-1].feature_names_in_
coef_ = log_reg[-1].coef_.ravel()

coef_series = pd.Series(coef_, index=feature_names).sort_values()

coef_series.plot(kind="barh");
