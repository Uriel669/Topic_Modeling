import requests as req
import pandas as pd 
import aiohttp , asyncio
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

def get_pages():
    pages = [f'https://www.fmprc.gov.cn/mfa_eng/xwfw_665399/s2510_665401/2511_665403/index_{i}.html' for i in range(1, 13, 1)]
    pages.insert(0 , 'https://www.fmprc.gov.cn/mfa_eng/xwfw_665399/s2510_665401/2511_665403/')
    return pages

def get_links(url):
    request = req.get(url)
    soup = BeautifulSoup(request.content, 'html.parser')
    li = soup.find('div' , {"class":"newsLst_mod"}).find('ul').find_all('li')
    suffix_list = [a.find('a')['href'] for a in li]
    suffix_list = [suffix[2:] for suffix in suffix_list]

    links = [f'https://www.fmprc.gov.cn/mfa_eng/xwfw_665399/s2510_665401/2511_665403/{url_suffix}' for url_suffix in suffix_list]
    return links

def get_text(soup):
    bare_text = [p.get_text() for p in soup.find_all('p')]
    bare_text = list(filter(None, bare_text))
    return bare_text

def get_date(soup):
    title = soup.find("h2", {"class": "title"}).get_text()
    word_list = title.split(' ')
    date_list = word_list[-3:]
    date_string = ' '.join(date_list)
    date = datetime.strptime(date_string , '%B %d, %Y')
    return str(date.strftime('%Y-%m-%d'))


def get_data():
    data = []
    for page in tqdm(get_pages()):
        for link in get_links(page):
            request = req.get(link)
            soup = BeautifulSoup(request.content, 'html.parser')
            date = get_date(soup)
            for text in get_text(soup):
                data.append([link , date , text])
    df = pd.DataFrame(data , columns=['link' , 'date' ,'text'])
    return df


async def get_data_fast(link):
    data = []
    async with aiohttp.ClientSession() as session:
        async with session.get(link , ssl=False) as response:
            html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        date = get_date(soup)
        for text in get_text(soup):
            data.append([link , date , text])
    return data

async def async_scrape():
    pages = get_pages()
    links = [get_links(page) for page in pages]
    all_links = [item for sublist in links for item in sublist]
    print(len(all_links))
    tasks = [asyncio.ensure_future(get_data_fast(link)) for link in all_links]
    total_data = []
    for task in asyncio.as_completed(tasks):
        print(task)
        data = await task
        total_data.append(data)
    df = pd.DataFrame(data , columns=['link' , 'date' ,'text'])
    print(df)
    return df
