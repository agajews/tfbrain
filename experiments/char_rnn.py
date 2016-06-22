import tfbrain as tb

from tasks.char_rnn import load_data


class LangModel(tb.UnhotXYModel):

    def build_net(self):
        self.num_cats = self.hyperparams['vocab_size']
        l_in = tb.ly.InputLayer((None,
                                 None,
                                 self.num_cats))
        net = l_in

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

        net = tb.ly.FullyConnectedLayer(net, 1024,
                                        nonlin=tb.nonlin.identity)

        net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_cats,
                                        nonlin=tb.nonlin.softmax)
        self.net = net
        self.input_vars = {'text': l_in.placeholder}


def train_char_rnn():
    print("Building network ...")

    gen_info = load_data(seqlength=20,
                         text_fnm='data/nietzsche.txt')
    train_xs = {'text': gen_info['data']['train']['text']}
    train_y = gen_info['data']['train']['targets']
    test_xs = {'text': gen_info['data']['test']['text']}
    test_y = gen_info['data']['test']['targets']
    print(train_xs['text'][:2])
    print(train_y[:2])

    hyperparams = {'batch_size': 128,
                   'learning_rate': 0.001,
                   'num_updates': 50000,
                   'grad_norm_clip': 5,
                   'vocab_size': gen_info['vocab_size'],
                   'char_to_index': gen_info['char_to_index'],
                   'index_to_char': gen_info['index_to_char'],
                   'seqlength': gen_info['seqlength']}

    model = LangModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.Perplexity(hyperparams)
    evaluator = tb.SeqGenEvaluator(hyperparams, loss, acc,
                                   seed='the quick brown fox jumps')
    optim = tb.AdamOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    trainer.train(train_xs, train_y, test_xs, test_y,
                  display_interval=100)

if __name__ == '__main__':
    train_char_rnn()
