import json
import numpy as np

with open('data/ticks.log') as f:
    rows = f.read().split('\n')[1:-1]

for row_num, row in enumerate(rows):
    rows[row_num] = json.loads(row)

exchange = '1'
rows = [row for row in rows if row['exchangeid'] == exchange]

num_ticks = 100
ticks = np.zeros((len(rows) - num_ticks, num_ticks, 2))
targets = np.zeros((len(rows) - num_ticks, 2))
for row_num, row in enumerate(rows[num_ticks:]):
    for tick_num, tick in enumerate(rows[row_num - num_ticks:row_num]):
        ticks[row_num, tick_num, 0] = tick['price']
        ticks[row_num, tick_num, 1] = tick['quantity']
    target_ind = 1 if row['price'] > rows[row_num - 1]['price'] else 0
    targets[row_num, target_ind] = 1

data_fnm = 'data/trading-systems.npz'
np.savez(data_fnm, ticks=ticks, targets=targets)
