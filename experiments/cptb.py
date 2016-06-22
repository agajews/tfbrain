import tfbrain as tb

from tasks.ptb import load_data


class PTBBasic(tb.UnhotYModel):

    def build_net(self):
        vocab_size = self.hyperparams['vocab_size']
        self.num_cats = vocab_size
        i_text = tb.ly.InputLayer(shape=(None, None),
                                  dtype=tb.int32)
        net = tb.ly.EmbeddingLayer(i_text, 200, vocab_size)
        # net = i_text
        net = tb.ly.LSTMLayer(
            net, 512)

        net = tb.ly.DropoutLayer(net, 0.5)

        # net = tb.ly.BasicRNNLayer(
        #     l_in, 200)

        # net = tb.ly.BasicRNNLayer(
        #     l_in, 200)

        net = tb.ly.LSTMLayer(
            net, 512)

        net = tb.ly.SeqSliceLayer(net, col=-1)
        # net = tb.ly.FlattenLayer(net)

        # net = tb.ly.FullyConnectedLayer(net, 1024,
        #                                 nonlin=tb.nonlin.identity)

        net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_cats,
                                        nonlin=tb.nonlin.softmax)
        self.net = net
        self.input_vars = {'text': i_text.placeholder}


def train_brnn():
    # gen_info = load_data(seqlength=20,
    #                      text_fnm='data/ptb/ptb.train.txt')
    # train_xs = {'text': gen_info['data']['train']['text']}
    # train_y = gen_info['data']['train']['targets']
    # val_xs = {'text': gen_info['data']['test']['text']}
    # val_y = gen_info['data']['test']['targets']
    # print(train_xs['text'][:2])
    # print(train_y[:2])
    # hyperparams = {'batch_size': 128,
    #                'learning_rate': 0.01,
    #                'num_updates': 50000,
    #                'grad_norm_clip': 5,
    #                'vocab_size': gen_info['vocab_size'],
    #                'char_to_index': gen_info['char_to_index'],
    #                'index_to_char': gen_info['index_to_char'],
    #                'seqlength': gen_info['seqlength']}

    seqlength = 20
    vocab_size = 10000
    data = load_data(seqlength=seqlength)
    train_xs = {'text': data['train']['text']}
    train_y = data['train']['targets']
    val_xs = {'text': data['test']['text']}
    val_y = data['test']['targets']

    hyperparams = {'batch_size': 128,
                   'learning_rate': 0.0001,
                   'num_updates': 50000,
                   'grad_norm_clip': 5,
                   'vocab_size': vocab_size,
                   'seqlength': seqlength}
    model = PTBBasic(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.Perplexity(hyperparams)
    evaluator = tb.PerplexityEvaluator(
        hyperparams, loss, acc,
        seed='the quick brown fox jumps')
    optim = tb.AdamOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    print('Training...')

    trainer.train(train_xs, train_y,
                  val_xs, val_y,
                  val_cmp=False,
                  display_interval=100)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train_brnn()
