import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import pandas as pd


# TODO remove 'varzesh3' from tokens
def return_dates():
    # Starting and ending dates in string format
    start_date = "1393/09/01"
    end_date = "1402/09/01"

    # Convert the string dates to datetime objects
    start = datetime.strptime(start_date, "%Y/%m/%d")
    end = datetime.strptime(end_date, "%Y/%m/%d")

    # List to store formatted dates
    formatted_dates = []

    # Generate dates and filter to include only specific days
    current_date = start
    while current_date <= end:
        # Include only days 01, 06, 11, 16, 21, 26
        if current_date.day in [1, 6, 11, 16, 21, 26]:
            formatted_dates.append(re.sub(r'/', '%2F', current_date.strftime("%Y/%m/%d")))

        # Move to the next month
        current_date += timedelta(days=1)  # Move one day forward

    # Print the list of formatted dates
    # print(formatted_dates)
    return formatted_dates


# construct the link which news of that particular day is stored in
def resolve_links():
    URLs = []
    dates = return_dates()
    # print( 'https://www.varzesh3.com/news?Date=' + dates[0])
    for date in tqdm(dates):
        address = 'https://www.varzesh3.com/news?Date=' + date
        page = requests.get(address)
        soup = BeautifulSoup(page.text, 'html.parser')
        # temp = (soup.select("body > section > main > div.container > div"))
        temp = soup.select('.newsbox-2 > a') # select all tags with class a with given attr.
        URLs.extend([tag['href'] for tag in temp[:4]])  # adding first 4 links of that day
        time.sleep(0.5)
    data = pd.DataFrame({
        'url': URLs,
        'tag': 's'
    })
    data.to_csv(r'D:\ca4\URLs.csv', index=False, encoding='utf-8')


##################################################
def extract_texts():
    data = pd.read_csv(r'D:\ca4\URLs.csv', encoding='utf-8')
    texts = []
    for link in tqdm(data['url']):
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        text = soup.select('.news-content')
        for tag in text[0].find_all('div', recursive=False):
            if tag is not None:
                if tag['class'][0] == 'news-main-detail':
                    main_detail = tag.select('h1')[0].text
                    caption = tag.select('p')[0].text
                    caption = re.sub(r'\s+', ' ', caption)
                    # print(main_detail)
                    # print(caption)
                elif tag['class'][0] == 'news-text':
                    news_text = []
                    for paragraphs in tag.select('p'):
                        if '\xa0' not in paragraphs.text:  # ignore <p class="rtejustify" style="text-align: justify;">&nbsp;</p>
                            # print(paragraphs.text)
                            news_text.append(paragraphs.text)
                            # print(news_text)
                    if not news_text:
                        # paragraphs = tag.select('div')[0].text
                        paragraphs_ = ''
                        paragraphs_ = tag.get_text(separator='\n', strip=True)
                        paragraphs_ = re.sub(r'\s+', ' ', paragraphs_)
                        if paragraphs_ == '':
                            news_text.append('')
                        else:
                            news_text.append(paragraphs_)
        if news_text:
            single_news = main_detail + caption + " ".join(news_text)
        else:
            single_news = main_detail + caption + paragraphs_
        texts.append(single_news)
        time.sleep(0.5)

    data['text'] = texts
    data.to_csv(r'D:\ca4\sport2.csv', index=False, encoding='utf-8')


def main():
    # links = resolve_links()
    extract_texts()


if __name__ == "__main__":
    main()
