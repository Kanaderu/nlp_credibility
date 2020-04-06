#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://euvsdisinfo.eu/disinformation-cases/?text=coronavirus&date='
print(url)

s = requests.Session()
s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/34.0.1847.116 Chrome/34.0.1847.116 Safari/537.36'
r = s.get(url)
if r.ok:
    print(r)


# In[2]:


soup = BeautifulSoup(r.content, 'html')
#print(soup.prettify()) # print the parsed data of html


# In[3]:


results_per_page = 10
num_results = soup.find('div', class_='disinfo-db-results').find('span').text
num_pages = int(np.ceil(float(num_results) / results_per_page))
print(f'There are {num_results} across {num_pages}')


# In[4]:


dataset = []
for page_num in range(num_pages):
    offset = page_num * results_per_page
    url = f'https://euvsdisinfo.eu/disinformation-cases/?text=coronavirus&date=&offset={offset}'
    r = s.get(url)
    if not r.ok:
        print(f'Unable to parse {url}')
        pass
    
    print(f'Parsing {url}')
    soup = BeautifulSoup(r.content, 'html')
    re_soup = soup.find_all('div', class_='disinfo-db-post')

    print(f'Found {len(re_soup)} tags at {url}')
    find_title = lambda res: res.find(attrs={'data-column': 'Title'}).text.strip()
    find_link = lambda res: res.find(attrs={'data-column': 'Title'}).find('a', href=True)['href']
    find_date = lambda res: res.find(attrs={'data-column': 'Date'}).text.strip()
    find_outlets = lambda res: res.find(attrs={'data-column': 'Outlets'}).text.strip()
    find_country = lambda res: res.find(attrs={'data-column': 'Country'}).text.strip()
    for idx, result in enumerate(re_soup):
        title = find_title(result)
        link = find_link(result)
        date = find_date(result)
        outlet = find_outlets(result)
        country = find_country(result)
        entry = [title, link, date, outlet, country]
        #print(entry)
        dataset.append(entry)


# In[5]:


df = pd.DataFrame(dataset, columns=['Title', 'Link', 'Date', 'Outlet', 'Country'])
display(df)


# In[6]:


for entry in dataset:
    article_url = entry[1]
    
    r = s.get(article_url)
    if not r.ok:
        print(f'Unable to parse {article_url}')
        pass
    
    print(f'Parsing {article_url}')
    soup = BeautifulSoup(r.content, 'html')
    summary = soup.find('div', class_='b-report__summary-text').text.strip()
    disproof = soup.find('div', class_='b-report__disproof-text').text.strip()
    
    article_title = soup.find('h1', class_='b-catalog__report-title').text.strip()
    
    # get the article source link
    try:
        article_source_link = soup.find('div', class_='b-catalog__link').find('a')['href']
        article_source_media = soup.find('div', class_='b-catalog__link').text.strip().replace(' (Archived', '')
    except:
        article_source_link = ''
        article_source_media = ''
    #print(article_source_link)
    #print(article_source_media)
    
    # get the optional article metadata
    metadata_list = dict()
    meta_list = soup.find('ul', class_='b-catalog__repwidget-list').find_all('li')
    for metadata in meta_list:
        extracted_label = metadata.find('b').extract().text.strip()
        d = metadata.text.strip()
        metadata_list[extracted_label] = d

    #print(metadata_list)
    
    reported_in =      metadata_list['Reported in:'] if 'Reported in:' in metadata_list else ''
    publication_date = metadata_list['DATE OF PUBLICATION:'] if 'DATE OF PUBLICATION:' in metadata_list else ''
    target_audience =  metadata_list['Language/target audience:'] if 'Language/target audience:' in metadata_list else ''
    country =          metadata_list['Country:'] if 'Country:' in metadata_list else ''
    keywords =         metadata_list['Keywords:'] if 'Keywords:' in metadata_list else ''
    
    #print(article_source_link, reported_in, publication_date, target_audience, country, keywords)
    
    article_entry = [summary, disproof, article_title, article_source_link, article_source_media, reported_in, publication_date, target_audience, country, keywords]
    entry.extend(article_entry)


# In[7]:


df = pd.DataFrame(dataset, columns=['Title', 'Link', 'Date', 'Outlet', 'Country', 'Summary', 'Disproof', 'Article Title', 'Article Source', 'Article Media', 'Reported In', 'Publication Date', 'Audience', 'Article Country', 'Keywords'])
display(df)
df.to_csv('euvsdisinfo_results.csv', index=False)


# In[8]:


article_data_cols = ['Summary', 'Disproof', 'Article Title', 'Article Source', 'Article Media', 'Reported In', 'Publication Date', 'Audience', 'Article Country', 'Keywords']
df_articles = df[article_data_cols]
display(df_articles)

df_articles.to_csv('euvsdisinfo_articles.csv', index=False)

