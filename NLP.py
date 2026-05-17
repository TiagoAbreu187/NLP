import re
import nltk
nltk.download('all')

import spacy

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud
from unidecode import unidecode
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk import word_tokenize
from nltk.corpus import stopwords

from spacy.lang.pt.stop_words import STOP_WORDS



# ----------------------------limpar texto---------------------------------------

reviews = pd.read_csv("NLP/am_scrape_final.csv")
print(reviews["Review"])

def limpar_texto(text):

    text = text.lower()
    text = re.sub(r'@[^\s]+', '', text)
    text = unidecode(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(www\.[^\s]+)|(https?://[^\s]+)', '', text)
    text = ''.join(c for c in text if not c.isdigit())
    text = ''.join(c for c in text if c not in punctuation)

    return word_tokenize(text)

texto_limpo = reviews["Review"].apply(limpar_texto)
print(texto_limpo)

# ----------------------no stopwords----------------------------------

sw = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))


def remove_stop_words(texts, stopwords=sw):

  new_texts = list()

  for word in texts:
    if word not in stopwords:
      new_texts.append(''.join(word))

  return new_texts

reviews["clean_text"] = reviews["Review"].apply(limpar_texto)

reviews["no_stopwords"] = reviews["clean_text"].apply(lambda text: [word for word in text if word not in sw])

print(reviews["no_stopwords"].iloc[0])


# --------------------lema-----------------------


nlp = spacy.load('pt_core_news_sm')

def verificar_lema(texto):
    doc = nlp(texto)
    
    for token in doc:
        print(f"{token.text:15} | {token.lemma_:15} | {token.pos_}")


print(verificar_lema(reviews.loc[0, "Review"]))

# ---------------------radical-----------------------

def verificar_radical(texto):
    stemmer = nltk.stem.SnowballStemmer('portuguese')
    tokens = word_tokenize(texto)

    for word in tokens:
        print(f"{word:15} | {stemmer.stem(word)}")

print(verificar_radical(reviews.loc[0, "Review"]))

# ------------------nuvem de palavras--------------------

def nuvem_palavras(textos):

  
    todas_palavras = ' '.join([texto for texto in textos])
   
    nuvem_palvras = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(todas_palavras)
  
    plt.figure(figsize=(24,12))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.show()

df = reviews["Review"]
nuvem_palavras(df)

# -------------------------bag of words--------------------------

texto = reviews["no_stopwords"].apply(lambda x: ' '.join(x))

vetorizar = CountVectorizer(max_features=1000,min_df=2,max_df=0.8)

bag_of_words = vetorizar.fit_transform(texto)

df_bow = pd.DataFrame(bag_of_words.toarray(),columns=vetorizar.get_feature_names_out())

print(df_bow.head())


# -------------------------sentimento-------------------------

X = df_bow
y = reviews["Sentiment"] = [1 if i % 2 == 0 else 0 for i in range(len(reviews))]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
