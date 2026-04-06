import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer, mo, pd, plt, re, stopwords


@app.cell
def _(pd):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train.info()
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
    test.drop(['location'], axis=1, inplace=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Распределение метрики у нас в целом равномерное, но для этой задачи нам лучше использовать метрику *Recall*. Нам нужна такая модель, которая будет с меньшей ошибкой выдавать прогноз по катастрофическим случаям - False Negative (модель с ошибкой прогнозировала отсуствие катастрофы, когда она явно случилась)

    Датасет состоит из 3 столбцов:

    1) keyword - категориальный признак, который нужно обработать (общая длина уникальных keyword составляет 222 (OHE не подходит)). Порядок категорий не важен (LE не подходит), поэтому будем использовать frequence и target encoding

    2) для текста будем использовать tf-idf и text_features (catboost)

    3) таргет y - числовой признак

    Помимо этого добавим еще несколько столбцов, которые могут помочь построить более точную модель:

    1) text_len - количество символов в сообщении

    2) word_count - количество слов в сообщении

    3) unique_rate - отношение уникальных слов ко всем словам в сообщении

    Таким образом занимаемся доработкой всех столбцов перед обучением
    """)
    return


@app.cell
def _(re, test, train):
    def cleaning(text):
        text=text.lower()
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are"
        }
        for contraction, full_form in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', full_form, text)

        text = re.sub(r'&amp;', ' ', text)      # &amp; → пробел
        text = re.sub(r'@\w+', ' ', text)       # упоминания → пробел
        text = re.sub(r'#', ' ', text)          # хэштеги → пробел
        text = re.sub(r'http\S+|www.\S+', ' ', text)  # ссылки → пробел

        # 2. ПОТОМ удаляем все остальные спецсимволы
        # (оставляем буквы, цифры, пробелы)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 3. Нормализуем пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    train['text'] = train['text'].apply(cleaning)
    test['text'] = test['text'].apply(cleaning)
    return


@app.cell
def _(pd, stopwords, train):
    #посмотрим самые популярные слова для обычной и катастр ситуации, а также уберем стоп-слова
    stopword_eng = set(stopwords.words('english'))
    basic_text = train[train['target']==0]['text']
    disaster_text = train[train['target']==1]['text']

    basic_text=basic_text.str.lower().str.split().explode()
    disaster_text=disaster_text.str.lower().str.split().explode()

    basic_text = basic_text[~basic_text.isin(stopword_eng)]
    disaster_text = disaster_text[~disaster_text.isin(stopword_eng)]

    top_basic = basic_text.value_counts().head(20).index
    top_disaster = disaster_text.value_counts().head(20).index

    pd.DataFrame({
        "Обычная ситуация": top_basic,
        "Катастрофическая ситуация": top_disaster
    })
    return (stopword_eng,)


@app.cell
def _(WordNetLemmatizer, stopword_eng, test, train):
    #произведем лемматизацию слов
    lemmatizer = WordNetLemmatizer()
    def predprocess_tweet(text):
        tokens=text.split()
        processed_token=[]
        for token in tokens:
            if token not in stopword_eng:
                lemmatized=lemmatizer.lemmatize(token)
                processed_token.append(lemmatized)

        return ' '.join(processed_token)
    train['text']=train['text'].apply(predprocess_tweet)
    test['text']=test['text'].apply(predprocess_tweet)

    train['text_len']=train['text'].str.len()
    train['word_count']=train['text'].str.split().str.len()
    train['unique_word_rate']=train['text'].str.split().apply(lambda x: len(set(x)) / (len(x) + 1))

    test['text_len']=test['text'].str.len()
    test['word_count']=test['text'].str.split().str.len()
    test['unique_word_rate']=test['text'].str.split().apply(lambda x: len(set(x)) / (len(x) + 1))

    train_filt = train[train['word_count']>= 4]
    train_filt.head()
    return (train_filt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Подготовка данных завершена и можно перейти к созданию baseline
    """)
    return


@app.cell
def _(test, train_filt):
    test.to_csv('data/test_ready.csv', index=False, encoding='utf-8')
    train_filt.to_csv('data/train_ready.csv', index=False, encoding='utf-8')
    return


if __name__ == "__main__":
    app.run()
