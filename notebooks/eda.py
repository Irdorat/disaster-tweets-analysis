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
    import nltk

    return mo, pd, plt


@app.cell
def _(pd):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train.head()
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Проверяем на пропуски данных, а также заменяем NaN на None.
    Удаляем столбец location, поскольку он создает лишний шум и не окажется полезным в предикте результата.
    Сделаем гистограмму признака keyword, чтобы понять есть ли выбросы и общая доля None
    """)
    return


@app.cell
def _(plt, test, train):
    #Заменяем NaN на None
    train['keyword'] = train['keyword'].fillna('None', inplace=False)
    test['keyword'] = test['keyword'].fillna('None', inplace=False)
    #удаляем столбец location и id - не несет полезной нагрузки на прогноз метки
    train.drop(['id', 'location'], axis=1, inplace=True)
    test.drop(['id','location'], axis=1, inplace=True)
    #строим гистограмму распределения признака keyword
    plt.hist(train['keyword'], bins=100, color='navy', edgecolor= 'black')
    plt.xlabel('Список keyword')
    plt.ylabel('Количество слов')
    plt.show()
    print(f"Количество строк в таблице, где keyword = None: {len(train[train['keyword']=='None'])}")
    print(f"Доля None относительно всех keyword: {len(train[train['keyword']=='None'])/len(train['keyword'])}")
    #оставляем столбец keyword
    return


@app.cell
def _(plt, train):
    #Далее смотрим на распределение таргета
    basic, disaster = train['target'].value_counts()
    basic, disaster
    plt.figure()
    labels = 'Basic', 'Disaster'
    size = [basic, disaster]
    plt.pie(
        size,
        labels=labels,
        shadow=False,
        autopct='%0.00f%%')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
