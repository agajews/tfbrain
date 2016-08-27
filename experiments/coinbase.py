import csv
from pprint import pprint
from datetime import datetime as dt, timedelta as td
import numpy as np

with open('data/coinbaseUSD.csv') as f:
    rows = list(csv.reader(f))

for row_num, row in enumerate(rows):
    rows[row_num] = {'ts': int(row[0]),
                     'price': float(row[1]),
                     'qty': float(row[2])}


def group_ticks_by(ticks, delta):
    time = dt.fromtimestamp(ticks[0]['ts'] - ticks[0]['ts'] % delta.seconds)
    print(time)
    groups = [{'time': time, 'ticks': []}]
    for tick in ticks:
        ticktime = dt.fromtimestamp(tick['ts'])
        if time <= ticktime <= time + delta:
            groups[-1]['ticks'].append(tick)
        elif time + delta <= ticktime <= time + 2 * delta:
            time = time + delta
            groups.append({'ticks': [tick],
                           'time': time})
        else:
            time = time + delta
            groups.append({'ticks': [],
                           'time': time})
    return groups

num_ticks = 1000

hours = group_ticks_by(rows, td(hours=1))
hours = [h for h in hours if len(h['ticks']) >= num_ticks]

print('Num hours: %d' % len(hours))
pprint(hours[:10])
pprint(hours[-10:])

max_num_ticks = max([len(hour['ticks']) for hour in hours])
min_num_ticks = min([len(hour['ticks']) for hour in hours])
print('Max ticks: %d' % max_num_ticks)
print('Min ticks: %d' % min_num_ticks)


def calc_trend(hour):
    return (hour['ticks'][-1]['price'] -
            hour['ticks'][0]['price'])


def calc_volume(hour):
    return sum(tick['qty'] for tick in hour['ticks'])


def calc_count(hour):
    return len(hour['ticks'])

data_hours = []
for hour_num, hour in enumerate(hours[:-1]):
    trend = calc_trend(hours[hour_num + 1])
    data_hours.append({
        'target': 1 if trend > 0 else 0,
        'first_ticks': hour['ticks'][:num_ticks],
        'last_ticks': hour['ticks'][-num_ticks:],
        'volume': calc_volume(hour),
        'trend': calc_trend(hour),
        'count': calc_count(hour),
    })

first_ticks = np.zeros((len(data_hours), num_ticks, 2))
last_ticks = np.zeros((len(data_hours), num_ticks, 2))
features = np.zeros((len(data_hours), 3))
targets = np.zeros((len(data_hours), 2))

for hour_num, hour in enumerate(data_hours):
    for tick_num, tick in enumerate(hour['first_ticks']):
        first_ticks[hour_num, tick_num, 0] = tick['price']
        first_ticks[hour_num, tick_num, 1] = tick['qty']

    for tick_num, tick in enumerate(hour['last_ticks']):
        first_ticks[hour_num, tick_num, 0] = tick['price']
        first_ticks[hour_num, tick_num, 1] = tick['qty']

    features[hour_num, 0] = hour['volume']
    features[hour_num, 1] = hour['trend']
    features[hour_num, 2] = hour['count']

    targets[hour_num, hour['target']] = 1

data_fnm = 'data/coinbase_n1000.npz'
np.savez(data_fnm, first_ticks=first_ticks, last_ticks=last_ticks,
         features=features, targets=targets)
