configuration = dict()
configuration['001'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "max_name_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,    # number of question per batch
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "relation_encoder": "ADD",
    "pattern_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
    "extra_keys": []
}

configuration['002'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "max_name_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "relation_encoder": "ADD",
    "pattern_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
    "topic_encoder": "RNN",
    "topic_config": {
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": False
    },
    "extra_keys": []
}


configuration['003'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "max_name_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "relation_encoder": "ADD",
    "pattern_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
    "extra_keys": ['entity_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['004'] = {
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "max_name_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "relation_encoder": "ADD",
    "pattern_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
    "topic_encoder": "RNN",
    "topic_config": {
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": False
    },
    "extra_keys": ['entity_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['005'] = {
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_sentence_len": 33,
    "max_word_len": 22,
    "max_name_len": 22,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "relation_encoder": "ADD",
    "pattern_config": {
        "word_dim": 50
    },
    "relation_config": {
        "word_dim": 50
    },
    "topic_encoder": "RNN",
    "topic_config": {
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": True
    },
    "extra_keys": ['entity_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}
