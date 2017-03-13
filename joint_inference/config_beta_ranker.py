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
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,    # number of question per batch
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
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
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_encoder": "RNN",
    "topic_config": {
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": False,
        "encoder": "RNN"
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
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD",
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD",
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
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": False,
        "encoder": "RNN"
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
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        "use_repr": True
    },
    "extra_keys": ['entity_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['006'] = {
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker",
    "fn_dev": "../data/ranker/wq.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/wq.simple.sub.rel.list.v3",
    "fn_relation": "../data/wq.simple.rel.list.v3",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 50,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        # "word_filter_sizes": [3],
        # "word_num_filters": 25,
        "word_bidirect": False,
        "use_repr": False,
    },
    "extra_keys": ['entity_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['007'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.aqqu.train.ranker",
    "fn_dev": "../data/ranker/wq.aqqu.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['008'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.aqqu.train.ranker",
    "fn_dev": "../data/ranker/wq.aqqu.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.9,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['009'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.aqqu.train.ranker",
    "fn_dev": "../data/ranker/wq.aqqu.test.ranker",
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.8,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['010'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.8,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['011'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.7,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['012'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [10, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.8,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'relation_score', 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['013'] = {
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [1],
    "activations": ["sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.8,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap']
}

configuration['014'] = {
    # relation is cast-actor
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [1],
    "activations": ["sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.8,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap']
}

configuration['015'] = {
    # relation is cast-actor, use lengths mask
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [1],
    "activations": ["sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", "relation_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap']
}

configuration['016'] = {
# relation is cast-actor, use lengths mask
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 33,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['017'] = {
    # relation is cast-actor, use lengths mask
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['018'] = {
    # relation is cast-actor, use lengths mask
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [100, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.0005,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 25,  # this is char-rnn
        "word_rnn_dim": 25,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'qw_rel_occur', 'num_answer']
}

configuration['019'] = {
    # relation is cast-actor, use lengths mask
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker.tmp",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.tmp",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}



configuration['020'] = {
    # relation is cast-actor, use lengths mask
    # type include notable type and type
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
    },
    "type_config": {
        "word_dim": 50,
        # "use_repr": False,
    },
    "question_config": {
        "encoder": "ADD",
        "word_dim": 50
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['021'] = {
    # base
    # F1 : 0.483344955176
    # Number of test case: 2015
    # Average rank: 6.3364764268
    # Average number of candidates: 55.0357320099
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    # "topic_config": {
    #     "encoder": "RNN",
    #     "word_dim": 50,  # this is char-rnn
    #     "word_rnn_dim": 50,
    #     "word_bidirect": False,
    #     # "word_filter_sizes": [3],
    #     # "word_num_filters": 50,
    #     "use_repr": False,
    # },

    "extra_keys": []
}

configuration['022'] = {
    # base + topic + mention-match
    # F1 : 0.477290880555
    # Number of test case: 2015
    # Average rank: 6.17518610422
    # Average number of candidates: 55.0357320099
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": []
}

configuration['023'] = {
    # base + topic + mention-match + constraint_entity_in_q
    # F1 : 0.482776078335
    # Number of test case: 2015
    # Average rank: 6.03225806452
    # Average number of candidates: 55.0357320099
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": ["constraint_entity_in_q"]
}

configuration['024'] = {
    # base + topic + mention-match + constraint_entity_in_q + constraint-entity-word-overlap
    # F1 : 0.480249234006
    # Number of test case: 2015
    # Average rank: 5.93846153846
    # Average number of candidates: 55.0357320099
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": ["constraint_entity_in_q", 'constraint_entity_word']
}

configuration['025'] = {
    # base + answer_type
    # F1 : 0.479725240464
    # Number of test case: 2015
    # Average rank: 5.83970223325
    # Average number of candidates: 54.9781637717
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_answer_type": '../data/ranker/answer.type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 50,
        "encoder": "ADD"
    },
    "answer_type_config": {
        "dim": 25
    },
    # "topic_config": {
    #     "encoder": "RNN",
    #     "word_dim": 50,  # this is char-rnn
    #     "word_rnn_dim": 50,
    #     "word_bidirect": False,
    #     # "word_filter_sizes": [3],
    #     # "word_num_filters": 50,
    #     "use_repr": False,
    # },

    "extra_keys": []
}

configuration['026'] = {
    # relation is cast-actor, use lengths mask
    # use 100 dimension word embeddings
    # F1 : 0.486012131336
    # Number of test case: 2015
    # Average rank: 5.96625310174
    # Average number of candidates: 54.9781637717
    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['027'] = {
    # relation is cast-actor, use lengths mask
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    # "topic_config": {
    #     "encoder": "RNN",
    #     "word_dim": 50,  # this is char-rnn
    #     "word_rnn_dim": 50,
    #     "word_bidirect": False,
    #     # "word_filter_sizes": [3],
    #     # "word_num_filters": 50,
    #     "max_pool": False,
    #     "use_repr": False,
    # },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['028'] = {
    # base + answer_type (question word VS type)
    # F1 : 0.479663531997
    # Number of test case: 2015
    # Average rank: 5.88039702233
    # Average number of candidates: 54.9781637717
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_answer_type": '../data/ranker/answer.type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "answer_type_config": {
        "word_dim": 50,
    },
    # "topic_config": {
    #     "encoder": "RNN",
    #     "word_dim": 50,  # this is char-rnn
    #     "word_rnn_dim": 50,
    #     "word_bidirect": False,
    #     # "word_filter_sizes": [3],
    #     # "word_num_filters": 50,
    #     "use_repr": False,
    # },

    "extra_keys": []
}


configuration['029'] = {
    # relation is cast-actor, use lengths mask
    # use 100 dimension word embeddings without "type config"
    # F1 : 0.484151890826
    # Number of test case: 2015
    # Average rank: 6.17518610422
    # Average number of candidates: 54.9781637717
    "gpu": 1,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },

    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['030'] = {
    # base + answer_type (pattern VS type) + all other features
    # F1 : 0.482239696865
    # Number of test case: 2015
    # Average rank: 5.67593052109
    # Average number of candidates: 54.9781637717
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_answer_type": '../data/ranker/answer.type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "answer_type_config": {
        "word_dim": 50,
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },

    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['031'] = {
    # base + answer_type (question word VS type) + all other features
    # F1 : 0.47794212806
    # Number of test case: 2015
    # Average rank: 6.13449131514
    # Average number of candidates: 54.9781637717
    "gpu": 0,
    "fn_train": "../data/ranker/wq.train.ranker",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_type": '../data/type.list',
    "fn_answer_type": '../data/ranker/answer.type.list',
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_question_len": 20,
    # 'max_type_len': 5,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.05,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "answer_type_config": {
        "word_dim": 50,
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "use_repr": False,
        "max_pool": True
    },

    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer']
}

configuration['032'] = {
    # relation is cast-actor, use lengths mask
    # use 100 dimension word embeddings

    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker.tmp",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.tmp",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer', 'gender_consistency']
}

configuration['033'] = {
    # relation is cast-actor, use lengths mask
    # use 100 dimension word embeddings

    "gpu": 2,
    "fn_train": "../data/ranker/wq.train.ranker.prs",  # crf mention tagger
    "fn_dev": "../data/ranker/wq.test.ranker.prs",     # crf mention tagger
    "fn_word": "../data/wq.simple.word.list.v3",  # TODO: generate new word list
    "fn_char": '../data/relation.char.list',      # TODO: generate new character list
    "fn_sub_relation": "../data/WebQSP/WebQSP.wq.simple.sub.rel.list",
    "fn_relation": "../data/WebQSP/WebQSP.wq.simple.rel.list",
    "hidden_layer_sizes": [50, 1],
    "activations": ["sigmoid", "sigmoid"],
    "max_pattern_len": 15,
    "max_word_len": 22,
    "max_name_len": 30,
    "reload": False,
    "num_epoch": 300,
    "batch_size": 20,
    "dropout_keep_prob": 0.5,
    "embedding_l2_scale": 0.,
    "l2_scale": 0.,
    "lr": 0.001,
    "optimizer": 'adam',
    "margin": 0.1,
    "pattern_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "relation_config": {
        "word_dim": 100,
        "encoder": "ADD"
    },
    "topic_config": {
        "encoder": "RNN",
        "word_dim": 50,  # this is char-rnn
        "word_rnn_dim": 50,
        "word_bidirect": False,
        # "word_filter_sizes": [3],
        # "word_num_filters": 50,
        "max_pool": False,
        "use_repr": False,
    },
    "extra_keys": ["entity_score", 'constraint_entity_in_q', 'constraint_entity_word', 'rel_pat_overlap', 'num_answer', 'gender_consistency', "type_consistency"]
}