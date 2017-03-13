

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

configuration['word-pe100-old'] = {
    "gpu": 2,
    "dir_name": 'word-add',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 0.9,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "use_position": True
    },
    "relation_config": {
        "word_dim": 100,
        "use_position": True
    }
}

configuration['word-cnn100-012'] = {
    "fn_train": "../data/WebQSP/WebQSP.crf.relation.train",
    "fn_dev": "../data/WebQSP/WebQSP.crf.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "encode_name": "CNN",
    "max_sentence_len": 15,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 1,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "word_filter_sizes": [3],
        "word_num_filters": 100
    },
    "relation_config": {
        "word_dim": 100,
        "word_filter_sizes": [3],
        "word_num_filters": 100
    }
}

configuration['word-cnn50-001'] = {
    "fn_train": "../data/WebQSP/WebQSP.crf.relation.train",
    "fn_dev": "../data/WebQSP/WebQSP.crf.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "encode_name": "CNN",
    "max_sentence_len": 15,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 1,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [2, 3],
        "word_num_filters": 25
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [2, 3],
        "word_num_filters": 25
    }
}

configuration['word-rnn100-011'] = {
    "dir_name": 'word-add',
    "fn_train": "../data/WebQSP/WebQSP.crf.relation.train",
    "fn_dev": "../data/WebQSP/WebQSP.crf.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "encode_name": "RNN",
    "max_sentence_len": 15,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 400,
    "batch_size": 50,
    "dropout_keep_prob": 1,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "word_bidirect": False,
        'word_rnn_dim': 100,
        # "use_position": True,
        "max_pool": True
    },
    "relation_config": {
        "word_dim": 100,
        "word_bidirect": False,
        "word_rnn_dim": 100,
        # "use_position": True,
        "max_pool": True
    }
}

configuration['word-add50-001'] = {
    "dir_name": 'word-add',
    "fn_train": "../data/WebQSP/WebQSP.crf.relation.train",
    "fn_dev": "../data/WebQSP/WebQSP.crf.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 0.5,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "use_position": False
    },
    "relation_config": {
        "word_dim": 50,
        "use_position": False
    }
}

configuration['word-add100-aqqu'] = {
    "dir_name": 'word-add',
    "fn_train": "../data/WebQSP/WebQSP.aqqu.relation.train",
    "fn_dev": "../data/WebQSP/WebQSP.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 0.9,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "use_position": False
    },
    "relation_config": {
        "word_dim": 100,
        "use_position": False
    }
}

configuration['word-add-50-exact-d0.9-101-tanh-adam-ranker'] = {
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [101, 1],
    "activations": ["tanh", "tanh"],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    'optimizer': 'adam',
    "margin": 0.1,
    "question_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
}


configuration['word-rnn-50-exact-d0.9-101-adam-ranker'] = {
    # "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_train": "../data/wq.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [101, 1],
    "activations": ["tanh", "tanh"],
    "encode_name": "RNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    'optimizer': 'adam',
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

configuration['word-cnn-50-exact-d0.9-101-adam-ranker'] = {
    "gpu": 2,
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    # "fn_train": "../data/wq.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [1],
    "activations": ["tanh"],
    "encode_name": "CNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    'optimizer': 'adam',
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [1],
        "word_num_filters": 50
    }
}

configuration['word-add-50-exact-d0.9-101-l2-adam-ranker'] = {
    "gpu": 2,
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [101, 1],
    "activations": ['sigmoid', 'sigmoid'],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.000000001,
    "l2_scale": 0.00001,
    "lr": 0.001,
    "optimizer": "adam",
    "margin": 0.1,
    "question_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    }
}


configuration['word-add-200-exact-d0.9-100-50-ranker'] = {
    "gpu": 2,
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 50, 1],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.01,
    "margin": 0.1,
    "question_config": {
        "word_dim": 200
    },
    "relation_config": {
        "word_dim": 200
    }
}

configuration['word-add-50-exact-d0.5-100-ranker'] = {
    # normalize question and relation repr
    "gpu": 2,
    # "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_train": "../data/wq.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 100,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "question_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    }
}

configuration['word-add-50-exact-d0.7-100-ranker'] = {
    # normalize question and relation repr
    "gpu": 2,
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 100,
    "dropout_keep_prob": 0.7,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "question_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    }
}

configuration['word-cnn-50-d0.9-100-ranker'] = {
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "CNN",
    "hidden_layer_sizes": [100, 1],
    "activations": ["tanh", "tanh"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 100,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.005,
    "optimizer": 'adam',
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    },
    "relation_config": {
        "word_dim": 50,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50
    }
}

configuration['word-add-100-exact-d0.9-100-50-ranker'] = {
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 50, 1],
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}




configuration['word-add-100-exact-d0.9-v2'] = {
    "dir_name": 'word-add-100-exact-d0.9-v2',
    "fn_train": "../data/merge_data/relation.train",
    "fn_dev": "../data/merge_data/wq.relation.test",
    "fn_word": "../data/merge_data/word.list",
    "fn_sub_relation": "../data/merge_data/sub.relation.list",
    "fn_relation": "../data/merge_data/relation.list",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": True,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}



configuration['word-add-100-exact-d0.9-v4'] = {
    # randomly choose negative relations
    "dir_name": 'word-add-100-exact-d0.9-v4',
    "gup": 2,
    "fn_train": "../data/my_fb/relation.train.tmp",
    "fn_dev": "../data/my_fb/wq.relation.test",
    "fn_word": "../data/merge_data/word.list",
    "fn_sub_relation": "../data/merge_data/sub.relation.list",
    "fn_relation": "../data/merge_data/relation.list",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}

configuration['word-add-100-exact-d0.9-v5'] = {
    # only randomly choose negative relations when negative relation set is empty
    "dir_name": 'word-add-100-exact-d0.9-v5',
    "gup": 2,
    "fn_train": "../data/my_fb/relation.train.tmp",
    "fn_dev": "../data/my_fb/wq.relation.test",
    "fn_word": "../data/word.list",
    "fn_sub_relation": "../data/merge_data/sub.relation.list",
    "fn_relation": "../data/merge_data/relation.list",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}

configuration['word-add-100-exact'] = {
    "dir_name": 'word-add-100-exact',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": True,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}

configuration['word-add-100-exact-d0.6'] = {
    "gpu": 1,
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 50,
    "dropout_keep_prob": 0.6,
    "lr": 0.001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100
    },
    "relation_config": {
        "word_dim": 100
    }
}

configuration['word-add-150-exact-d0.9'] = {
    "dir_name": 'word-add-150-exact-d0.9',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.9,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 150
    },
    "relation_config": {
        "word_dim": 150
    }
}

configuration['word-add-200-exact-d0.8'] = {
    "dir_name": 'word-add-200-exact-d0.8',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "ADD",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 0.8,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 200
    },
    "relation_config": {
        "word_dim": 200
    }
}

configuration['word-cnn-50'] = {
    "fn_train": "../data/wq.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "CNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 100,
    "batch_size": 100,
    "dropout_keep_prob": 0.8,
    "lr": 0.0005,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    }
}

configuration['word-cnn-50-2'] = {
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
    "batch_size": 100,
    "dropout_keep_prob": 0.8,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [3],
        "word_num_filters": 50
    }
}

configuration['word-cnn-50-3'] = {
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
    "batch_size": 100,
    "dropout_keep_prob": 0.8,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 50,
        "word_filter_sizes": [3, 4],
        "word_num_filters": 25
    },
    "relation_config": {
        "word_dim": 50,
        "word_filter_sizes": [2, 3],
        "word_num_filters": 25
    }
}

configuration['word-cnn-150'] = {
    "dir_name": 'word-cnn-150',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "CNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 150,
        "word_filter_sizes": [3],
        "word_num_filters": 150
    },
    "relation_config": {
        "word_dim": 150,
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


configuration['word-rnn-200'] = {
    "dir_name": 'word-rnn-200',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "RNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 100,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 200,
        "word_bidirect": False,
        "word_rnn_dim": 200
    },
    "relation_config": {
        "word_dim": 200,
        "word_bidirect": False,
        "word_rnn_dim": 200
    }
}

configuration['word-rnn-200-2'] = {
    "dir_name": 'word-rnn-200-2',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "RNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 100,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 200,
        "word_bidirect": False,
        "word_rnn_dim": 200
    },
    "relation_config": {
        "word_dim": 200,
        "word_bidirect": False,
        "word_rnn_dim": 200
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
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 50,
    "batch_size": 50,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
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

configuration['word-rnn-add'] = {
    "dir_name": 'word-rnn-add',
    "fn_train": "../data/wq.simple.aqqu.relation.train",
    "fn_dev": "../data/wq.aqqu.relation.test",
    "fn_char": '../data/relation.char.list',
    "fn_word": "../data/wq.simple.word.list.v3",
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "encode_name": "RNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "word_bidirect": False,
        "word_rnn_dim": 200
    },
    "relation_config": {
        "word_dim": 200,
    }
}

configuration['word-rnn-100-v2'] = {
    "dir_name": 'word-rnn-100-v2',
    "fn_train": "../data/merge_data/relation.train",
    "fn_dev": "../data/merge_data/wq.relation.test",
    "fn_word": "../data/merge_data/word.list",
    "fn_sub_relation": "../data/merge_data/sub.relation.list",
    "fn_relation": "../data/merge_data/relation.list",
    "encode_name": "RNN",
    "max_sentence_len": 33,
    "max_word_len": 22,
    "reload": False,
    "num_epoch": 200,
    "batch_size": 100,
    "dropout_keep_prob": 1,
    "lr": 0.0001,
    "margin": 0.1,
    "question_config": {
        "word_dim": 100,
        "word_bidirect": False,
        "word_rnn_dim": 100
    },
    "relation_config": {
        "word_dim": 100,
        "word_bidirect": False,
        "word_rnn_dim": 100
    }
}