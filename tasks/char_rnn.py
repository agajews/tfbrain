import numpy as np

from .ptb import indices_to_seq_data


def get_indices(text, char_to_index):
    return list(map(lambda c: char_to_index[c], text))


def load_data(text_fnm='data/nietzsche.txt',
              seqlength=20,
              split=0.25):

    with open(text_fnm) as f:
        text = f.read()
    chars = list(set(text))
    vocab_size = len(chars)
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}

    split_ind = int(len(text) * split)
    train_indices = get_indices(text[:split_ind], char_to_index)
    test_indices = get_indices(text[split_ind:], char_to_index)

    train_data = indices_to_seq_data(train_indices, seqlength)
    test_data = indices_to_seq_data(test_indices, seqlength)
    data = {'train': train_data,
            'test': test_data}

    return {'data': data,
            'char_to_index': char_to_index,
            'index_to_char': index_to_char,
            'seqlength': seqlength,
            'vocab_size': vocab_size}


def gen_sequence(model,
                 seqlength,
                 char_to_index,
                 index_to_char,
                 seed='The quick brown fox jumps',
                 num_chars=100):

    assert len(seed) >= seqlength
    assert len(model.input_vars) == 1
    samples = []
    indices = get_indices(seed, char_to_index)
    data = np.zeros((1, seqlength))
    input_name = list(model.input_vars.keys())[0]
    for i, index in enumerate(indices[:seqlength]):
        data[0, i] = index

    for i in range(num_chars):
        # Pick the character that got assigned the highest probability
        preds = model.compute_preds({input_name: data})
        ix = np.argmax(preds.ravel())
        # Alternatively, to sample from the distribution instead:
        # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
        samples.append(ix)
        data[0, 0:seqlength - 1] = data[0, 1:]  # bump down
        data[0, seqlength - 1] = ix  # insert latest

    random_snippet = seed + ''.join(
        index_to_char[index] for index in samples)
    return random_snippet
