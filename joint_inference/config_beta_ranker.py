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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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
    "max_sentence_len": 33,
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