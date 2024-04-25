import numpy as np
import pandas as pd
import random
def combine_all_data():
    collected_sport = pd.read_csv(r'D:\ca4\sport\sport2.csv', encoding='utf-8')
    collected_political = pd.read_csv(r'D:\ca4\politics\politics.csv', encoding='utf-8')
    given_dataset = pd.read_csv(r'D:\ca4\given\nlp_train.csv', encoding='utf-8')

    all_texts = []
    all_texts.append(collected_sport['text'].tolist())
    all_texts.append(collected_political['text'].tolist())
    all_texts.append(given_dataset['Text'].tolist())

    combined_texts = [string for sublist in all_texts for string in sublist]

    all_tags = []
    all_tags.append(collected_sport['tag'].tolist())
    all_tags.append(collected_political['tag'].tolist())
    all_tags.append(given_dataset['Category'].tolist())

    combined_tags = [tag for sublist in all_tags for tag in sublist]

    return combined_texts,combined_tags


def store_all_data(texts,tags):
    ds = pd.DataFrame({
        'text': texts,
        'tag': tags
    })
    ds.to_csv(r'D:\ca4\all.csv', index=False, encoding='utf-8-sig')


def main():
    all_data,all_tags = combine_all_data()
    store_all_data(all_data,all_tags)



if __name__ == "__main__":
    main()
