import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from prophet import Prophet
from bs4 import BeautifulSoup
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from datetime import date



start_date = '2017-01-01'
end_date = date.today()
type_ids = open('type_ids.txt', 'r')
items = {}
time_steps = 180

for i in type_ids.readlines():
    items[int(i.split('|')[0])] = i.split('|')[1].split('\n')[0]
    
for type_id in items.keys():
    if type_id < 0:
        continue
    ds = []
    y = []
    print(items[type_id], ' ', type_id)
    session = requests.Session()
    page = session.get(f'https://www.adam4eve.eu/commodity.php?regionID=10000002&typeID={type_id}&from={start_date}&until={end_date}&avg=7')
    soup = BeautifulSoup(page.content, 'html.parser')
    if len(soup.find_all('script')) < 9:
        continue
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
    try:
        m.fit(df)
        future = m.make_future_dataframe(periods = time_steps)

        forecast = m.predict(future)

        dp = [0 for i in range(len(forecast))]
        datedp = [0 for i in range(len(forecast))]
        dp[len(forecast) - 1] = forecast['yhat'][len(forecast) - 1]
        datedp[len(forecast) - 1] = forecast['ds'][len(forecast) - 1]
        for i in reversed(range(len(forecast) - time_steps, len(forecast) - 1)):
            if dp[i + 1] > forecast['yhat'][i]:
                dp[i] = dp[i + 1]
                datedp[i] = datedp[i + 1]
            else:
                dp[i] = forecast['yhat'][i]
                datedp[i] = forecast['ds'][i]

        yhats = [forecast['yhat'][i] for i in range(len(forecast))]
        max_difference = dp[len(forecast) - time_steps] - yhats[len(forecast) - time_steps]
        datel = forecast['ds'][len(forecast) - time_steps]
        dater = datedp[len(forecast) - time_steps]
        ratio = 0

        for i in range(len(forecast) - time_steps + 1, len(forecast)):
            difference = dp[i] - yhats[i]
            if(difference > max_difference):
                max_difference = difference
                ratio = max_difference / yhats[i] * 100
                datel = forecast['ds'][i]
                dater = datedp[i]

        mape = 0
        if ratio >= 40:
            df_cv = cross_validation(m, initial='1095 days', period = '30 days', horizon = f'{time_steps} days', parallel = 'processes')
            df_p = performance_metrics(df_cv)
            for i in df_p['mape']:
                mape += i

            mape /= len(df_p['mape'])
            mape *= 100
            file = open('profitable_items.txt', 'a')
            file.write(f'{items[type_id]} {type_id} | profit: {ratio}% | buy: {str(datel).split()[0]} | sell: {str(dater).split()[0]} | mape: {mape}%')
            file.write('\n')
            file.close()
            print('mape: ', mape)
        print(f'{items[type_id]} {type_id} | profit: {ratio}% | buy: {str(datel).split()[0]} | sell: {str(dater).split()[0]}')
        print('-----------')
    except ValueError:
        print('error')
