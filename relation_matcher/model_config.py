

configuration = dict()

configuration['word-add'] = {
    "dir_name": 'word-add',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": True,
    "num_epoch": 50,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    }
}

configuration['word-cnn'] = {
    "dir_name": 'word-cnn',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "CNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 100,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 150,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    },
    "relation_config": {
        "word_dim": 150,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    }
}

configuration['char-cnn'] = {
    "dir_name": 'char-cnn',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_char": '../data/relation.char.list',
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "CNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 50,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "char_dim": 50,
        "word_filter_sizes": [2, 3],
        "word_num_filters": 25,
        "char_filter_sizes": [2, 3],
        "char_num_filters": 25
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [2, 3],
        "word_num_filters": 25
    }
}

configuration['word-rnn'] = {
    "dir_name": 'word-rnn',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "RNN",
    "max_sentence_len": 36,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 50,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_bidirect": False,
        "word_rnn_dim": 50
    },
    "relation_config": {
        "word_dim": 50,
        "word_bidirect": False,
        "word_rnn_dim": 50
    }
}

configuration['word-birnn'] = {
    "dir_name": 'word-birnn',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "RNN",
    "max_sentence_len": 36,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 50,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_bidirect": True,
        "word_rnn_dim": 50
    },
    "relation_config": {
        "word_dim": 50,
        "word_bidirect": True,
        "word_rnn_dim": 50
    }
}

