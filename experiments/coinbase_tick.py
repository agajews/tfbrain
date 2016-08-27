import csv
import numpy as np

with open('data/coinbaseUSD.csv') as f:
    rows = list(csv.reader(f))[-100000:]

for row_num, row in enumerate(rows):
    rows[row_num] = {'ts': int(row[0]),
                     'price': float(row[1]),
                     'qty': float(row[2])}

num_ticks = 100
ticks = np.zeros((len(rows) - num_ticks, num_ticks, 2))
targets = np.zeros((len(rows) - num_ticks, 2))
for row_num, row in enumerate(rows[num_ticks:]):
    if row_num % 10000 == 0:
        print(row_num)
    for tick_num, tick in enumerate(rows[row_num - num_ticks:row_num]):
        ticks[row_num, tick_num, 0] = tick['price']
        ticks[row_num, tick_num, 1] = tick['qty']
    target_ind = 1 if row['price'] > rows[row_num - 1]['price'] else 0
    targets[row_num, target_ind] = 1

data_fnm = 'data/coinbase-ticks-100000.npz'
np.savez(data_fnm, ticks=ticks, targets=targets)
