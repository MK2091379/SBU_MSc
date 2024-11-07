categorical_cols = ['thal', 'ca','slope', 'exang', 'restecg', 'fbs', 'cp' , 'sex', 'num']
bool_cols = ['fbs' , 'exang']
numeric_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'age']
imputer = IterativeImputer(random_state=42, max_iter=10)
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

test[numeric_cols] = test[numeric_cols].round(0)
train[numeric_cols] = train[numeric_cols].round(0)
for column in categorical_cols[:-1]:  # Ignore the target column 'num'
    most_frequent_value = train[column].mode()[0]
    train[column] = train[column].fillna(most_frequent_value)
    test[column] = test[column].fillna(most_frequent_value)
train['chol'] = train['chol'].replace(0, np.nan)
test['chol'] = test['chol'].replace(0, np.nan)
imputer = IterativeImputer(random_state=42, max_iter=10)
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

test[numeric_cols] = test[numeric_cols].round(0)
train[numeric_cols] = train[numeric_cols].round(0)
def cap_outliers(df, cols):
    for col in cols:
        upper_limit = df[col].mean() + 3 * df[col].std()
        lower_limit = df[col].mean() - 3 * df[col].std()
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
    return df

train = cap_outliers(train, numeric_cols)
scaler = MinMaxScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])
# Encoding Categorical Variables
train = pd.get_dummies(train, columns=categorical_cols[:-1], drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols[:-1], drop_first=True)

# Align test data with train data
test = test.reindex(columns=train.columns, fill_value=0)
test = test.drop(columns='num')
train[train.select_dtypes(['bool']).columns] = train.select_dtypes(['bool']).astype(int)
test[test.select_dtypes(['bool']).columns] = test.select_dtypes(['bool']).astype(int)
X = train.drop(columns=['id', 'dataset', 'num'])
y = train['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'SVC': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

best_model = None
best_f1_score = 0
results = []
best_score = 0
train_accuracy = []
skf = StratifiedKFold(n_splits=5)

# Loop through models
for name, model in models.items():
    print(f'Tuning hyperparameters for {name}...')
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[name],
                                           n_iter=50, scoring='f1_macro', cv=skf, verbose=1, n_jobs=-1)
    
    randomized_search.fit(X_train, y_train)
    best_estimator = randomized_search.best_estimator_
    print(f'Best hyperparameters for {name}: {randomized_search.best_params_}')

    # Evaluate the best model
    y_pred_train = best_estimator.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    y_pred = best_estimator.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({
        'Model': name,
        'Accuracy': accu,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1

    })
    train_accuracy.append({
        'Model': name,
        'Training Accuracy': train_acc
    })

    if accu > best_score:
        best_score = accu
        best_model = name
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = name

print(f'\nTraining completed. Best model: {best_model} with F1 score: {best_f1_score:.2f}')

model = KNeighborsClassifier()
model.fit(X, y)
datat = test.drop(columns = ['id', 'dataset'])
predictions = model.predict(datat)
submission_df = pd.DataFrame({
    'ID': test.index,
    'num': predictions
})
submission_df.to_csv('submit.csv', index = False)
