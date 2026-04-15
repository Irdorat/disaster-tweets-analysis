import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re
    from nltk.corpus import stopwords
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, precision_recall_curve, make_scorer, average_precision_score
    import pickle
    import os
    import category_encoders as ce

    return (
        ColumnTransformer,
        GridSearchCV,
        LogisticRegression,
        Pipeline,
        TfidfVectorizer,
        accuracy_score,
        auc,
        average_precision_score,
        ce,
        confusion_matrix,
        f1_score,
        make_scorer,
        mo,
        np,
        pd,
        plt,
        precision_recall_curve,
        precision_score,
        re,
        recall_score,
        roc_curve,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Согласно соревнованию нам нужно использовать метрику F1

    В качетсве baseline модели будем использовать модель логистической регрессии

    Начнем с кодировки категориальных признаков
    """)
    return


@app.cell
def _(pd):
    train = pd.read_csv('data/train_ready.csv')
    validation = pd.read_csv('data/test_ready.csv')
    #Заменяем NaN на None
    train['keyword'] = train['keyword'].fillna('None', inplace=False)
    validation['keyword'] = validation['keyword'].fillna('None', inplace=False)
    train = train.dropna(subset=['text'])
    train.info()
    #train_nan_rows = train[train['text'].isna()]
    #print(train_nan_rows)
    #train_loc=train.loc[4497]
    #print(train_loc)
    return (train,)


@app.cell
def _(train):
    train[train['target']==0].tail(10)
    return


@app.cell
def _(
    ColumnTransformer,
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    ce,
    f1_score,
    make_scorer,
    train,
    train_test_split,
):
    #Разделение данных
    X = train.drop(columns = ['target'])
    y = train['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=17,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1,2), #учитывает слова отдельные слова, пары и словосочетания до 3 слов
        max_features=5000, #кол-во частотных признаков
        min_df=3, #игнорирует слова, которые встречаются менее чем в 5 твитах
        sublinear_tf=True
    )

    keyword_col = ce.TargetEncoder()
    preprocessor=ColumnTransformer(
        transformers=[
            ('text', tfidf, 'text'),
            ('keyword', keyword_col, 'keyword')
        ]
    )

    # Используем F1 как метрику
    f1_scorer = make_scorer(f1_score)

    # Сетка параметров
    param_grid = {
        'classifier__C': [0.05, 0.1, 0.3],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear'],
        'classifier__class_weight': [
            'balanced',              # автоматический расчет: 1/(57%) = 1.75 для класса 1
            {0: 1, 1: 1.3},         # ручной вес (57/43 ≈ 1.325)
        ]
    }

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=3000,
            random_state=17
        ))
    ])

    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший F1 на кросс-валидации:", grid_search.best_score_)

    model = grid_search.best_estimator_
    return X_test, X_train, model, y_test, y_train


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    model,
    np,
    pd,
    y_test,
    y_train,
):
    y_train_predict = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_predict = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.8, 0.01)
    best_thresh = 0.5
    best_f1_test = 0
    best_f1_train = 0
    best_score = 0

    for thresh in thresholds:
        y_train_pred = (y_train_proba >= thresh).astype(int)
        y_test_pred = (y_test_proba >= thresh).astype(int)

        f1_train = f1_score(y_train, y_train_pred)
        f1_test = f1_score(y_test, y_test_pred)
        diff = abs(f1_train - f1_test)

        score = f1_test

        # Штраф только если переобучение > 10%
        if diff > 0.1:
            penalty = 1 - (diff - 0.1)  # линейный штраф после порога
            score *= max(penalty, 0.7)  # не ниже 70% от исходного

        # Дополнительный бонус за низкое переобучение
        if diff < 0.05:
            score *= 1.05  # бонус 5%

        print(f"Threshold {thresh}: F1 train = {f1_train:.3f} F1 test = {f1_test:.3f}, где разница между трейном и тестом составляет: {abs(f1_train - f1_test):.3f} ({((f1_train - f1_test) / f1_train * 100):.1f}% переобучения), а score порога равен {score}")
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_f1_test = f1_test
            best_f1_train = f1_train
            best_diff = diff

    print(f'В качестве лучшего порога выбрано: {best_thresh}')
    y_train_predict = (y_train_proba >= best_thresh).astype(int)
    y_test_predict = (y_test_proba >= best_thresh).astype(int)

    result = pd.DataFrame(
        {'Метрика': ['Train','test'],
        'Accuracy': [accuracy_score(y_train, y_train_predict), accuracy_score(y_test,y_test_predict)],
        'F1-мера': [f1_score(y_train, y_train_predict), f1_score(y_test,y_test_predict)]
        }
    )
    print (result)
    return y_test_predict, y_test_proba, y_train_predict


@app.cell
def _(auc, plt, roc_curve, y_test, y_test_proba):
    #ROC-AUC
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_test_proba) #false positive rate (доля ложноположительных срабатываний), true positive rate (чувствительность recall)
    roc_auc = auc(fpr_logreg, tpr_logreg) #AUC = 1 — идеальная классификация, 0.5 — случайная

    plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc:.3f})')

    plt.plot([0,1],[0,1],'--') ## Эта линия соответствует случайной модели (AUC=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    confusion_matrix,
    pd,
    precision_score,
    recall_score,
    y_test,
    y_test_predict,
    y_train,
    y_train_predict,
):
    cm = confusion_matrix(y_test, y_test_predict) #матрица неточностей для теста, показывающая TP, FN, TN, FP
    cm_df = pd.DataFrame(
        cm,
        index=['Negative', 'Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    print(cm_df)
    print(f"Precission for train: {precision_score(y_train, y_train_predict):.3f}\nRecall for train: {recall_score(y_train, y_train_predict):.3f}\nPrecission for test: {precision_score(y_test, y_test_predict):.3f}\nRecall for test: {recall_score(y_test, y_test_predict):.3f}")
    return


@app.cell
def _(plt, precision_recall_curve, y_test, y_test_proba):
    precision, recall, t = precision_recall_curve(y_test, y_test_proba)

    plt.plot(t, precision[:-1], label='Precision')
    plt.plot(t, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    return precision, recall


@app.cell
def _(model, re):
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coef = model.named_steps['classifier'].coef_[0]

    top_pos = coef.argsort()[-5:]
    top_neg = coef.argsort()[:5]

    print("=== TOP слов при катастрофе ===")
    for i in top_pos:
        clean_name = re.sub(r'^text__', '', feature_names[i])
        print(f"{clean_name}: {coef[i]:.1f}")

    print("\n=== Топ слов опред обычную ситуацию ===")
    for i in top_neg:
        clean_name = re.sub(r'^text__', '', feature_names[i])
        print(f"{clean_name}: {coef[i]:.1f}")
    return


@app.cell
def _(average_precision_score, plt, precision, recall, y_test, y_test_proba):
    ap = average_precision_score(y_test, y_test_proba)

    # Строим PR кривую
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AP = {ap:.3f})')

    # Настройка графика
    plt.xlabel('Recall (Полнота)', fontsize=12)
    plt.ylabel('Precision (Точность)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
