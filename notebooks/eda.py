import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import numpy as np
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk

    return WordNetLemmatizer, mo, nltk, np, pd, plt, re, sns, stopwords, string


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
def _(re):
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

    return (cleaning,)


@app.cell
def _():
    return


@app.cell
def _(
    WordNetLemmatizer,
    cleaning,
    nltk,
    np,
    re,
    stopwords,
    string,
    test,
    train,
):
    stopword_eng = set(stopwords.words('english'))
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

    train['word_count']=train['text'].str.split().str.len()
    train['unique_word_rate']=train['text'].str.split().apply(lambda x: len(set(x)) / (len(x) + 1))
    train['stop_word_count'] = train['text'].apply(lambda x: (len([w for w in str(x).lower().split() if w in stopword_eng])))
    train['stop_word_ratio'] = train['stop_word_count'] / (train['word_count'] + 1)
    train['no_stop_list'] = train['text'].apply(lambda x: ' '.join([w for w in str(x).lower().split() if w not in stopword_eng]))
    train['no_stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w not in stopword_eng]))
    train['url_count'] = train['text'].apply(lambda x: (len([w for w in str(x).lower().split() if 'http' in w or 'https' in w])))
    train['mean_word_count'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    train['upper_count'] = train['text'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1))
    train['char_count'] = train['text'].apply(lambda x: (len(str(x))))
    train['punctuation_count'] = train['text'].apply(lambda x: (len([c for c in str(x) if c in string.punctuation])))
    train['hashtag_count'] = train['text'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    train['mention_count'] = train['text'].apply(lambda x: len(re.findall(r'@\w+', str(x))))
    train['digit_count'] = train['text'].apply(lambda x: len(re.findall(r'\d', str(x))))
    train['upper_char_count'] = train['text'].apply(lambda x: (len([c for c in str(x) if c.isupper()])))
    train['spec_symb'] = train['text'].apply(lambda x: (len([c for c in re.findall(r'[^\s\w]|_', x)])))
    train['exclam_count'] = train['text'].apply(lambda x: str(x).count('!'))
    train['quest_count'] = train['text'].apply(lambda x: str(x).count('?'))
    train['repeat_char_count'] = train['text'].apply(lambda x: len(re.findall(r'(.)\1{2,}', str(x).lower())))
    train['intensity_score'] = train['exclam_count'] + train['upper_count'] / (train['char_count'] + 1) + train['repeat_char_count']
    train['adjective_count'] = train['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('JJ')))
    train['adverb_count'] = train['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('RB')))
    train['verb_count'] = train['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('VB')))
    train['noun_count'] = train['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('NN')))
    train['text'] = train['text'].apply(cleaning)
    train['text']=train['text'].apply(predprocess_tweet)


    test['word_count']=test['text'].str.split().str.len()
    test['unique_word_rate']=test['text'].str.split().apply(lambda x: len(set(x)) / (len(x) + 1))
    test['stop_word_count'] = test['text'].apply(lambda x: (len([w for w in str(x).lower().split() if w in stopword_eng])))
    test['stop_word_ratio'] = test['stop_word_count'] / (test['word_count'] + 1)
    test['no_stop_list'] = test['text'].apply(lambda x: ' '.join([w for w in str(x).lower().split() if w not in stopword_eng]))
    test['no_stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w not in stopword_eng]))
    test['url_count'] = test['text'].apply(lambda x: (len([w for w in str(x).lower().split() if 'http' in w or 'https' in w])))
    test['mean_word_count'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test['upper_count'] = test['text'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1))
    test['char_count'] = test['text'].apply(lambda x: (len(str(x))))
    test['punctuation_count'] = test['text'].apply(lambda x: (len([c for c in str(x) if c in string.punctuation])))
    test['hashtag_count'] = test['text'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    test['mention_count'] = test['text'].apply(lambda x: len(re.findall(r'@\w+', str(x))))
    test['digit_count'] = test['text'].apply(lambda x: len(re.findall(r'\d', str(x))))
    test['upper_char_count'] = test['text'].apply(lambda x: (len([c for c in str(x) if c.isupper()])))
    test['spec_symb'] = test['text'].apply(lambda x: (len([c for c in re.findall(r'[^\s\w]|_', x)])))
    test['exclam_count'] = test['text'].apply(lambda x: str(x).count('!'))
    test['quest_count'] = test['text'].apply(lambda x: str(x).count('?'))
    test['repeat_char_count'] = test['text'].apply(lambda x: len(re.findall(r'(.)\1{2,}', str(x).lower())))
    test['intensity_score'] = test['exclam_count'] + test['upper_count'] / (test['char_count'] + 1) + test['repeat_char_count']
    test['adjective_count'] = test['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('JJ')))
    test['adverb_count'] = test['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('RB')))
    test['verb_count'] = test['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('VB')))
    test['noun_count'] = test['text'].apply(lambda x: sum(1 for word, pos in nltk.pos_tag(str(x).split()) if pos.startswith('NN')))
    test['text'] = test['text'].apply(cleaning)
    test['text']=test['text'].apply(predprocess_tweet)



    train_filt = train[train['word_count']>= 4]
    train_filt.head()
    return stopword_eng, train_filt


@app.cell
def _(train_filt):
    train_filt.describe()
    return


@app.cell
def _(np, sns, train_filt):
    train_heat = train_filt.select_dtypes(include=[np.number])
    sns.heatmap(train_heat.corr())
    return


@app.cell
def _(train_filt):
    train_filt.info()
    return


@app.cell
def _(train_filt):
    train_filt.head()
    return


@app.cell
def _(test, train_filt):
    train_heat2 = train_filt.drop(
        columns=[
            'char_count',
            'no_stop_word_count',
            'stop_word_count',
            'punctuation_count',
            'spec_symb',
            'upper_count',
            'upper_char_count'
        ]
    )

    test2 = test.drop(
        columns=[
            'char_count',
            'no_stop_word_count',
            'stop_word_count',
            'punctuation_count',
            'spec_symb',
            'upper_count',
            'upper_char_count'
        ]
    )
    return test2, train_heat2


@app.cell
def _(np, sns, train_heat2):
    train_heat3 = train_heat2.select_dtypes(include=[np.number])
    sns.heatmap(train_heat3.corr())
    return


@app.cell
def _(pd, stopword_eng, train):
    #посмотрим самые популярные слова для обычной и катастр ситуации, а также уберем стоп-слова
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Подготовка данных завершена и можно перейти к созданию baseline
    """)
    return


@app.cell
def _(test2, train_heat2):
    test2.to_csv('data/test_ready.csv', index=False, encoding='utf-8')
    train_heat2.to_csv('data/train_ready.csv', index=False, encoding='utf-8')
    return


if __name__ == "__main__":
    app.run()
