X, y = fetch_openml(data_id=1050, parser="pandas", as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0)

log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", penalty="l1")
)
log_reg.set_output(transform="pandas")

log_reg.fit(X_train, y_train);

log_reg.score(X_test, y_test)

coefs = log_reg[-1].coef_.flatten()
feature_names_in = log_reg[-1].feature_names_in_

coefs_series = pd.Series(coefs, index=feature_names_in).sort_values()

coefs_series.plot.barh(figsize=(12, 8));
