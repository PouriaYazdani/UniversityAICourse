import string
import pandas as pd
from hazm import Normalizer
from hazm import word_tokenizer
from hazm import Stemmer
from hazm import Lemmatizer


def read(path):
    return pd.read_csv(path, encoding='utf-8')


def normalize(data,text_col = 'Text'):
    normalizer = Normalizer(correct_spacing=True, remove_diacritics=True,
                            remove_specials_chars=True, decrease_repeated_chars=False,
                            persian_style=False, persian_numbers=True,
                            unicodes_replacement=True, seperate_mi=True)
    normalized_texts = []
    for text in data[text_col]:
        normalized_texts.append(normalizer.normalize(text))

    return normalized_texts


def tokenize(data):
    tokenizer = word_tokenizer.WordTokenizer(join_verb_parts=True, join_abbreviations=True,
                                             separate_emoji=False, replace_links=False,
                                             replace_ids=False, replace_emails=False,
                                             replace_numbers=True, replace_hashtags=False)

    tokenized_texts = []

    for text in data:
        tokenized_texts.append(tokenizer.tokenize(text))

    return tokenized_texts


def remove_redundnat(data):
    redundant = ['،', '؟', '؛', '»', '«', '!؟']
    for s in string.punctuation:
        redundant.append(s)
    for char in range(ord('a'), ord('z') + 1):
        redundant.append(chr(char))
    for char in range(ord('A'), ord('Z') + 1):
        redundant.append(chr(char))
    redundant.append('NUM')

    for i in range(10):
        redundant.append(str(i))

    removed = []
    for text in data:
        temp = []
        for word in text:
            if word not in redundant:
                temp.append(word)
        removed.append(temp)

    return removed


def stem(data):
    stemmer = Stemmer()

    stemmed = []
    for text in data:
        temp = []
        for word in text:
            temp.append(stemmer.stem(word))
        stemmed.append(temp)
    return stemmed




def remove_stopword(data):
    prepositional = ['سه','تابناک','به','که','در','با','از',
                 'بر','اثر','آن','ان','این','چنین','برای','طور','اینطور','جز','بنا','همین','هیج','بعد','سایر','هم','چه'
        ,'که','را','نیز','نه','تا','باتوجه','یا','و','ولی','اگر','بلکه','هر','اما']

    pronoun = ['من','تو','او','شما','ایشان','آن\u200cها','ما','وی']

    stopwords = prepositional + pronoun

    removed = []

    for text in data:
        temp = []
        for word in text:
            if word not in stopwords:
                temp.append(word)
        removed.append(temp)

    return removed


def lemmatize(data):
    lemmatizer = Lemmatizer()

    lemmatized = []
    for text in data:
        temp = []
        for word in text:
            temp.append(lemmatizer.lemmatize(word))
        lemmatized.append(temp)

    lemmatized_and_removed = []

    most_repeated_verbs = list(pd.read_csv(r'D:\ca4\utility\not important\verbs_num.csv', encoding='utf-8')['count verbs'][
                               0:12])  # more than 10000 occurrence

    for text in lemmatized:
        temp = []
        for word in text:
            if not any(word in s for s in most_repeated_verbs) :
                temp.append(word)
        lemmatized_and_removed.append(temp)


    return lemmatized_and_removed


def write(texts,tags):
    ds = pd.DataFrame({
        'text': texts,
        'tag' : tags
    })
    ds.to_csv(r'D:\ca4\cleaned.csv', index=False, encoding='utf-8')
def main():
    data = read(r'D:\ca4\utility\all.csv')
    normalized_data = normalize(data)
    tokenized_data = tokenize(normalized_data)
    removed_redundants = remove_redundnat(tokenized_data)
    lemmatized = lemmatize(removed_redundants)
    removed_stopwords = remove_stopword(lemmatized)
    write(removed_stopwords,data['Category'])

    a = 2



if __name__ == "__main__":
    main()
