import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from prophet import Prophet
from bs4 import BeautifulSoup


start_date = '2017-01-01'
end_date = '2022-08-28'
type_ids = open('type_ids.txt', 'r')
items = {}

for i in type_ids.readlines():
    items[int(i.split('|')[0])] = i.split('|')[1].split('\n')[0]
    
for type_id in items.keys():
    ds = []
    y = []

    session = requests.Session()
    page = session.get(f'https://www.adam4eve.eu/commodity.php?regionID=10000002&typeID={type_id}&from={start_date}&until={end_date}')
    soup = BeautifulSoup(page.content, 'html.parser')
    values = str(soup.find_all('script')[8])
    values = values.split('d = new Dygraph(document.getElementById("graphdiv"),')[0].split('{')[0]

    for i in range(values.count('new Date')):
        entry = values.split('[new Date(\'')[i + 1]
        date = entry.split('\'')[0]
        price = entry.split('[')[1].split(',')[1]
        ds.append(date)
        y.append(float(price))
    
    if not ds or not y:
        continue
    d = {'ds': ds, 'y': y}
    df = pd.DataFrame(data = d)

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods = 30)

    forecast = m.predict(future)

#     fig1 = m.plot(forecast)
#     fig2 = m.plot_components(forecast)

    maxdate = None
    maxprice = -100000000000000000000
    current_price = forecast['yhat'][len(forecast) - 31]

    for i in range(len(forecast) - 30, len(forecast)):
        if forecast['yhat'][i] > maxprice:
            maxprice = forecast['yhat'][i]
            maxdate = forecast['ds'][i]
        
    if (maxprice - current_price) / current_price * 100 >= 20:
        file = open('profitable_items.txt', 'a')
        file.write(f'{items[type_id]} {type_id}')
        file.write('\n')
        file.write(f'highest profit on {maxdate} | profit: {(maxprice - current_price) / current_price * 100}%')
        file.write('\n')
        file.close()
    print(type_id, ' ', (maxprice - current_price) / current_price * 100, '%')