# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:42:25 2016

@author: ZFang
"""
import concurrent.futures
from urllib.request import urlopen
import time
import pandas as pd


ticker = pd.ExcelFile('tickerlist.xlsx')
ticker_df = ticker.parse(str(ticker.sheet_names[0]))
ticker_list = list(ticker_df['Ticker'])

start = time.time()

        
        
        
# Retrieve a single page and report the url and contents
def load_url(ticker, timeout):
    url = 'http://finance.yahoo.com/quote/' + ticker
    conn = urlopen(url, timeout=timeout)
    return ticker,conn.read()

# We can use a with statement to ensure threads are cleaned up promptly

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    li = []
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, ticker, 60): ticker for ticker in ticker_list}
    for future in concurrent.futures.as_completed(future_to_url):
        url = 'http://finance.yahoo.com/quote/' + future_to_url[future]
        try:
            data = future.result()
            li.append(str(data))
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('"%s" fetched in %ss' % (url,(time.time() - start)))
print("Elapsed Time: %ss" % (time.time() - start))

