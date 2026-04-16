import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, make_scorer, average_precision_score, precision_recall_curve
    from scipy.sparse import hstack
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import pickle
    import os

    return (
        CatBoostClassifier,
        GridSearchCV,
        Pool,
        accuracy_score,
        auc,
        average_precision_score,
        confusion_matrix,
        f1_score,
        make_scorer,
        mo,
        np,
        pd,
        plt,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(pd):
    train = pd.read_csv('data/train_ready.csv')
    #Заменяем NaN на None
    train['keyword'] = train['keyword'].fillna('None', inplace=False)
    train = train.dropna(subset=['text'])
    train.head()
    return (train,)


@app.cell
def _(Pool, train, train_test_split):
    X = train.drop(columns = ['target'])
    y = train['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=17,
        stratify=y
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=['keyword'],
        text_features=['text']
    )
    test_pool = Pool(
        X_test,
        y_test, 
        cat_features=['keyword'],
        text_features=['text']
    )
    return X_test, X_train, test_pool, train_pool, y_test, y_train


@app.cell
def _(X_train):
    X_train.info()
    return


@app.cell
def _(
    CatBoostClassifier,
    GridSearchCV,
    X_train,
    f1_score,
    make_scorer,
    y_train,
):
    #Затратно и не запускать на тесте
    model = CatBoostClassifier(
        iterations = 1000,
        random_seed=17,
        verbose=100,
        thread_count=-1,
        task_type="GPU",
        devices="0",
        early_stopping_rounds=200
    )

    f1_scorer = make_scorer(f1_score)
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'auto_class_weights': ['Balanced', 'SqrtBalanced'],  # None = balanced автоматически? Нет, нужно указать
    }

    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=1,  
        verbose = 2
    )

    grid_search.fit(
        X_train, y_train,
        cat_features=['keyword'],
        text_features=['text']
    )

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший F1 на кросс-валидации:", grid_search.best_score_)

    model = grid_search.best_estimator_
    return


@app.cell
def _(CatBoostClassifier, test_pool, train_pool):
    model_catboost=CatBoostClassifier(
        iterations=7000,
        learning_rate=0.01,
        depth=10,
        l2_leaf_reg=7,
        auto_class_weights='Balanced',

        task_type="GPU",
        devices="0",

        early_stopping_rounds=200,
        verbose=100
    )

    model_catboost.fit(train_pool, eval_set=test_pool)
    return (model_catboost,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    model_catboost,
    np,
    pd,
    y_test,
    y_train,
):
    y_train_predict = model_catboost.predict(X_train)
    y_train_proba = model_catboost.predict_proba(X_train)[:, 1]
    y_test_predict = model_catboost.predict(X_test)
    y_test_proba = model_catboost.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.2, 0.8, 0.01)
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

        print(f"Threshold {thresh}: F1 train = {f1_train:.3f} F1 test = {f1_test:.3f}, где разница: {abs(f1_train - f1_test):.3f} ({((f1_train - f1_test) / f1_train * 100):.1f}% переобучения), а score равен {score}")
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
    return best_thresh, y_test_predict, y_test_proba, y_train_predict


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Выполним
    """)
    return


@app.cell
def _(pd):
    subm_csv = pd.read_csv('data/test_ready.csv')
    subm_csv.info()
    return (subm_csv,)


@app.cell
def _(subm_csv):
    subm_csv['keyword'] = subm_csv['keyword'].fillna('None', inplace=False)
    subm_csv['text'] = subm_csv['text'].fillna('None', inplace=False)
    subm_csv.info()
    return


@app.cell
def _(subm_csv):
    subm_csv_X = subm_csv.drop(['id'], axis=1)
    subm_csv_y = subm_csv[['id']]
    subm_csv_X.head()
    return subm_csv_X, subm_csv_y


@app.cell
def _(best_thresh, model_catboost, subm_csv_X):
    subm_csv_proba = model_catboost.predict_proba(subm_csv_X)[:,1]
    subm_csv_X['target'] = (subm_csv_proba >= best_thresh).astype(int)
    return


@app.cell
def _(subm_csv_X, subm_csv_y):
    subm_csv_X_index = subm_csv_X.reset_index()
    subm_csv_y_index = subm_csv_y.reset_index()
    subm_csv_merged = subm_csv_y_index.merge(
        subm_csv_X_index, 
        on='index', 
        how='left')
    subm_csv_merged = subm_csv_merged.drop('index', axis=1)
    subm_csv_merged = subm_csv_merged.drop(['keyword', 'text', 'text_len', 'word_count', 'unique_word_rate'], axis = 1)
    subm_csv_merged.to_csv('data/test_submission.csv', index=False, encoding='utf-8')
    subm_csv_merged.info(20)
    return


if __name__ == "__main__":
    app.run()
