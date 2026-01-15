# SMOTE, Scaler, PCA


smote =SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_val_scaled = scaler.transform(X_val)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_val_pca = pca.transform(X_val_scaled)


knn= KNeighborsClassifier()
params = [
    {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['cosine'],
        'algorithm': ['auto', 'brute'] 
    },
    {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=knn, 
    param_grid=params, 
    cv=skf, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_pca, y_res)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_}") 




y_pred =grid_search.predict(X_val_pca)

print(classification_report(y_val, y_pred, target_names=labels))
