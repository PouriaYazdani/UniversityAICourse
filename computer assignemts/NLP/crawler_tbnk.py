import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from dateutil.relativedelta import relativedelta

def extract_political():
    links = pd.read_csv(r'D:\ca4\politicalLinks.csv', encoding='utf-8')

    df = pd.DataFrame({
        'url': [],
        'tag': [],
        'text': []
    }
    )
    url = []
    text = []
    total = 2500
    num_news = 0
    progress_bar = tqdm(total=total, desc="Processing")
    # no_politics_limit = 500
    final = 0
    for link in links['links']:
        print(str(final) + "TH LINK")
        final += 1
    # link = links['links'][0]
        if num_news == total:
            break
        sitemap = requests.get(link)
        sitemap.encoding = 'utf-8'
        soup = BeautifulSoup(sitemap.text, 'lxml')
        all_links = ([('https://www.tabnak.ir'+news.get('href')) for news in soup.select(
            '#inner_night > div.container > div > div.col-md-32.gutter_main_inner > div > div.col-ms-28.gutter_inner'
            ' > div > div.archive_content')[0].find_all('a')])
        news_limit = 53 # 53 news for each link
        # no_politics_counter = 0
        for l in all_links:

            if num_news == total:
                break
            page = requests.get(l)
            page.encoding = 'utf-8'
            page_soup = BeautifulSoup(page.text, 'lxml')

            print(l)
            url.append(l)
            paragraphs = []
            text_body = page_soup.select('#newsMainBody')
            if text_body is not None:
                text_body = text_body[0]
                paragraphs = [re.sub(r'\s+', ' ',p.text) for p in  text_body.find_all('p')]
            title = page_soup.select('#news > div.container.night_mode_news > div > div.col-md-22.col-sm-24.gutter_news > div.top_news_title > div.title > h1')
            if title is not None:
                title = title[0].text
                paragraphs.append(title)
            subtitle = page_soup.select('#news > div.container.night_mode_news > div > div.col-md-22.col-sm-24.gutter_news > div.top_news_title > div.subtitle')
            if subtitle is None:
                subtitle = subtitle[0].getText()
                paragraphs.append(re.sub(r'\s+', ' ',subtitle))
            text.append(" ".join(paragraphs))
            # no_politics_counter -= 5

            num_news += 1
            progress_bar.update(1)

            news_limit -= 1
            print(text[-1])
    # no_politics_counter += 1
    # print(no_politics_counter)
        if news_limit == 0 :
            break


    print("final link is " + str(final))
    df['url'] = url
    df['text'] = text
    df['tag'] = 'p'
    df.to_csv(r'D:\ca4\politics.csv', index=False, encoding='utf-8')


def generate_date_tuples(start_date, end_date):
    date_tuples = []

    current_date = datetime.strptime(start_date, "%Y/%m/%d")
    end_date = datetime.strptime(end_date, "%Y/%m/%d")

    while current_date < end_date:
        next_date = current_date + relativedelta(months=1)  # Add one month
        date_tuples.append((current_date.strftime("%Y/%m/%d"), next_date.strftime("%Y/%m/%d")))
        current_date = next_date

    return date_tuples


def political_links():
    start_date = "1399/01/01"
    end_date = "1402/11/01"
    date_tuples = generate_date_tuples(start_date, end_date)

    base_url = "https://www.tabnak.ir/fa/archive?service_id=24&sec_id=-1&cat_id=-1&rpp=100"
    links = []

    for date_tuple in date_tuples:
        from_date, to_date = date_tuple
        link = f"{base_url}&from_date={from_date}&to_date={to_date}&p=1"
        links.append(link)

    data = pd.DataFrame({
        'links': links
    })
    data.to_csv(r'D:\ca4\politicalLinks.csv', index=False, encoding='utf-8')

# box-top-news > div.hidden-xs.col-md-12.col-sm-36.col-ms-14 > div > a:nth-child(1)
def main():
    # political_links()
    extract_political()

if __name__ == "__main__":
    main()
