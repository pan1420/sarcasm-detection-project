
# FINAL SARCASM DETECTION PROJECT (POLISHED)


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

 
# 1. LOAD DATA
 
def load_data(path):
    df = pd.read_csv(path)

    required_cols = ['headline', 'is_sarcastic']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df


 
# 2. FEATURE ENGINEERING
 
def add_features(df):
    analyzer = SentimentIntensityAnalyzer()

    df['length'] = df['headline'].apply(len)
    df['exclamation_count'] = df['headline'].apply(lambda x: x.count('!'))
    df['question_count'] = df['headline'].apply(lambda x: x.count('?'))

    df['sentiment'] = df['headline'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )

    return df

 
# 3. SPLIT DATA
 
def split_data(df):
    return train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_sarcastic']
    )



# 4. BUILD PIPELINE

def build_pipeline(model):
    text_feature = 'headline'
    numeric_features = ['length', 'exclamation_count', 'question_count', 'sentiment']

    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(stop_words='english'), text_feature),
        ('num', StandardScaler(), numeric_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

    return pipeline


 
# 5. HYPERPARAMETER TUNING
 
def tune_model(pipeline, X_train, y_train):
    param_grid = {
        'preprocessor__text__max_features': [3000, 5000, 10000],
        'preprocessor__text__ngram_range': [(1,1), (1,2)],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    return grid.best_estimator_


 
# 6. TRAIN & EVALUATE
 
def train_and_evaluate(train_df, test_df):
    models = {
        "SVM": LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        print(f"\n===== Training {name} =====")

        pipeline = build_pipeline(model)
        best_model = tune_model(pipeline, train_df, train_df['is_sarcastic'])

        y_pred = best_model.predict(test_df)

        acc = accuracy_score(test_df['is_sarcastic'], y_pred)
        print(f"{name} Accuracy: {acc * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(test_df['is_sarcastic'], y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(test_df['is_sarcastic'], y_pred))

        results[name] = (best_model, y_pred, acc)

    return results


 
# 7. CLEAN COMPARISON TABLE
 
def compare_all_models(test_df, results):
    print("\n===== FINAL MODEL COMPARISON =====")

    comparison = []

    # Add YOUR models
    for name, (_, _, acc) in results.items():
        comparison.append((name + " (Your Model)", acc))

    # Add external AI models
    ai_models = ['Openai_Sarcasm', 'DeepSeek_Sarcasm', 'Gemini_Sarcasm', 'llama_Sarcasm']

    for ai_name in ai_models:
        if ai_name in test_df.columns:
            acc = accuracy_score(test_df['is_sarcastic'], test_df[ai_name])
            comparison.append((ai_name, acc))

    # Sort by accuracy
    comparison.sort(key=lambda x: x[1], reverse=True)

    print("\nModel Performance Ranking:\n")
    print(f"{'Model':<30} | {'Accuracy':>10}")
    print("-" * 45)

    for model_name, acc in comparison:
        print(f"{model_name:<30} | {acc * 100:>9.2f}%")

    print("\n🏆 Best Model:", comparison[0][0])



# 8. ERROR ANALYSIS

def error_analysis(test_df, y_pred):
    print("\n===== ERROR ANALYSIS =====")

    errors = test_df.copy()
    errors['pred'] = y_pred

    mistakes = errors[errors['is_sarcastic'] != errors['pred']]

    for _, row in mistakes.head(5).iterrows():
        print("\nHeadline:", row['headline'])
        print("Actual:", row['is_sarcastic'], "| Predicted:", row['pred'])



# 9. LIVE TEST

def live_test(model):
    analyzer = SentimentIntensityAnalyzer()

    print("\n===== LIVE TEST =====")

    while True:
        text = input("\nEnter headline (or 'exit'): ")
        if text.lower() == 'exit':
            break

        temp_df = pd.DataFrame({'headline': [text]})
        temp_df = add_features(temp_df)

        pred = model.predict(temp_df)[0]
        result = "SARCASTIC" if pred else "NOT SARCASTIC"

        print("Prediction:", result)



# MAIN

def main():
    print("=== FINAL SARCASM DETECTION PROJECT ===")

    df = load_data('sarcasm_dataset.csv')

    df = add_features(df)

    train_df, test_df = split_data(df)

    results = train_and_evaluate(train_df, test_df)

# Best model

    best_model_name = max(results, key=lambda x: results[x][2])
    best_model, best_preds, _ = results[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    compare_all_models(test_df, results)

    error_analysis(test_df.reset_index(drop=True), best_preds)

    live_test(best_model)


if __name__ == "__main__":
    main()






# # =========================================
# # ADVANCED SARCASM DETECTION PROJECT
# # =========================================

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# # =========================================
# # 1. LOAD DATA
# # =========================================
# def load_data(path):
#     df = pd.read_csv(path)

#     required_cols = ['headline', 'is_sarcastic']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Missing column: {col}")

#     return df


# # =========================================
# # 2. FEATURE ENGINEERING
# # =========================================
# def add_features(df):
#     analyzer = SentimentIntensityAnalyzer()

#     # Text-based features
#     df['length'] = df['headline'].apply(len)
#     df['exclamation_count'] = df['headline'].apply(lambda x: x.count('!'))
#     df['question_count'] = df['headline'].apply(lambda x: x.count('?'))

#     # Sentiment feature
#     df['sentiment'] = df['headline'].apply(
#         lambda x: analyzer.polarity_scores(x)['compound']
#     )

#     return df


# # =========================================
# # 3. SPLIT DATA
# # =========================================
# def split_data(df):
#     return train_test_split(
#         df,
#         test_size=0.2,
#         random_state=42,
#         stratify=df['is_sarcastic']
#     )


# # =========================================
# # 4. BUILD PIPELINE
# # =========================================
# def build_pipeline(model):
#     text_features = 'headline'
#     numeric_features = ['length', 'exclamation_count', 'question_count', 'sentiment']

#     preprocessor = ColumnTransformer([
#         ('text', TfidfVectorizer(stop_words='english'), text_features),
#         ('num', StandardScaler(), numeric_features)
#     ])

#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('clf', model)
#     ])

#     return pipeline


# # =========================================
# # 5. HYPERPARAMETER TUNING
# # =========================================
# def tune_model(pipeline, X_train, y_train):
#     param_grid = {
#         'preprocessor__text__max_features': [3000, 5000, 10000],
#         'preprocessor__text__ngram_range': [(1,1), (1,2)],
#     }

#     grid = GridSearchCV(
#         pipeline,
#         param_grid,
#         cv=3,
#         n_jobs=-1,
#         verbose=1
#     )

#     grid.fit(X_train, y_train)

#     print("\nBest Parameters:", grid.best_params_)
#     return grid.best_estimator_


# # =========================================
# # 6. TRAIN & EVALUATE
# # =========================================
# def train_and_evaluate(train_df, test_df):
#     models = {
#         "SVM": LinearSVC(class_weight='balanced', random_state=42),
#         "LogisticRegression": LogisticRegression(max_iter=1000)
#     }

#     results = {}

#     for name, model in models.items():
#         print(f"\n===== Training {name} =====")

#         pipeline = build_pipeline(model)
#         best_model = tune_model(pipeline, train_df, train_df['is_sarcastic'])

#         y_pred = best_model.predict(test_df)

#         acc = accuracy_score(test_df['is_sarcastic'], y_pred)
#         print(f"{name} Accuracy: {acc * 100:.2f}%")

#         print("\nClassification Report:")
#         print(classification_report(test_df['is_sarcastic'], y_pred))

#         print("Confusion Matrix:")
#         print(confusion_matrix(test_df['is_sarcastic'], y_pred))

#         results[name] = (best_model, y_pred, acc)

#     return results


# # =========================================
# # 7. FAIR COMPARISON
# # =========================================
# def compare_with_other_models(test_df):
#     print("\n===== FAIR COMPARISON =====")

#     ai_models = ['Openai_Sarcasm', 'DeepSeek_Sarcasm', 'Gemini_Sarcasm', 'llama_Sarcasm']

#     for ai_name in ai_models:
#         if ai_name in test_df.columns:
#             acc = accuracy_score(test_df['is_sarcastic'], test_df[ai_name])
#             print(f"{ai_name} Accuracy: {acc * 100:.2f}%")
#         else:
#             print(f"{ai_name} not found.")


# # =========================================
# # 8. ERROR ANALYSIS
# # =========================================
# def error_analysis(test_df, y_pred):
#     print("\n===== ERROR ANALYSIS =====")

#     errors = test_df.copy()
#     errors['pred'] = y_pred

#     mistakes = errors[errors['is_sarcastic'] != errors['pred']]

#     for _, row in mistakes.head(5).iterrows():
#         print("\nHeadline:", row['headline'])
#         print("Actual:", row['is_sarcastic'], "| Predicted:", row['pred'])


# # =========================================
# # 9. LIVE TEST
# # =========================================
# def live_test(model):
#     analyzer = SentimentIntensityAnalyzer()

#     print("\n===== LIVE TEST =====")

#     while True:
#         text = input("\nEnter headline (or 'exit'): ")
#         if text.lower() == 'exit':
#             break

#         temp_df = pd.DataFrame({'headline': [text]})
#         temp_df = add_features(temp_df)

#         pred = model.predict(temp_df)[0]
#         result = "SARCASTIC" if pred else "NOT SARCASTIC"

#         print("Prediction:", result)


# # =========================================
# # MAIN
# # =========================================
# def main():
#     print("=== ADVANCED SARCASM DETECTION PROJECT ===")

#     df = load_data('sarcasm_dataset.csv')

#     df = add_features(df)

#     train_df, test_df = split_data(df)

#     results = train_and_evaluate(train_df, test_df)

#     # Select best model
#     best_model_name = max(results, key=lambda x: results[x][2])
#     best_model, best_preds, _ = results[best_model_name]

#     print(f"\nBest Model: {best_model_name}")

#     compare_with_other_models(test_df)

#     error_analysis(test_df.reset_index(drop=True), best_preds)

#     live_test(best_model)


# # =========================================
# # RUN
# # =========================================
# if __name__ == "__main__":
#     main()