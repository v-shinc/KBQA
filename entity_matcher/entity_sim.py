
import tensorflow as tf
import json
import os
QUESTION_LEN = 10
DESCRIPTION_LEN = 20


def cosine(u, v):
    dot = tf.reduce_sum(tf.mul(u, v), 1)
    sqrt_u = tf.sqrt(tf.reduce_sum(u ** 2), 1)
    sqrt_v = tf.sqrt(tf.reduce_sum(v ** 2), 1)
    epsilon = 1e-5
    return dot / (tf.maximum(sqrt_u * sqrt_v), epsilon)

class EntitySim(object):
    def __init__(self, num_word, embed_size, margin, lr, load_path):
        self.questions = tf.placeholder(tf.int32, [None, QUESTION_LEN], name='questions')
        self.descriptions = tf.placeholder(tf.int32, [None, DESCRIPTION_LEN], name='descriptions')
        self.neg_descriptions = tf.placeholder(tf.int32, [None, DESCRIPTION_LEN], name='neg_descriptions')

        with tf.variable_scope("embedding"):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
            embeddings = tf.get_variable('embeddings', [num_word, embed_size], initializer=initializer)
            question_words = tf.nn.embedding_lookup(embeddings, self.questions)
            description_words = tf.nn.embedding_lookup(embeddings, self.descriptions)
            neg_description_words = tf.nn.embedding_lookup(embeddings, self.neg_descriptions)

        with tf.variable_scope('pooling'):
            question_sum = tf.reduce_sum(question_words, 1)
            description_sum = tf.reduce_sum(description_words, 1)
            neg_description_sum = tf.reduce_sum(neg_description_words, 1)

        with tf.variable_scope('dropout'):
            question_drop = tf.nn.dropout(question_sum, 0.5)
            description_drop = tf.nn.dropout(description_sum, 0.5)
            neg_description_drop = tf.nn.dropout(neg_description_sum, 0.5)

        with tf.variable_scope('loss'):
            self.pos_score = cosine(question_drop, description_drop)
            neg_score = cosine(question_drop, neg_description_drop)
            self.loss = tf.maximum(0., neg_score + margin - self.pos_score)

        opt = tf.train.AdamOptimizer(lr)
        self.train_op = opt.minmize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.InteractiveSessions(config=config)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        if load_path:
            self.saver.restore(self.session, load_path)
        else:
            self.session.run(tf.initialize_all_variables())

    def init_from_conf(self, filename):
        conf = json.load(filename, encoding='utf8')
        load_path = os.path.abspath(os.path.join(os.path.curdir, 'runs', conf['model_name'], 'checkpoints', 'model'))
        return EntitySim(conf['num_word'], conf['embed_size'], conf['margin'], conf['lr'], load_path)

    def predict(self, questions, descriptions):
        scores = self.session.run(
            self.pos_score,
            {self.questions: questions,
             self.descriptions: descriptions}
        )
        return scores

    def fit(self, questions, descriptions, neg_descriptions):
        _, loss = self.session.run([self.train_op, self.loss],{
            self.questions: questions,
            self.descriptions: descriptions,
            self.neg_descriptions: neg_descriptions
        })
        return loss

    def save(self, save_path):
        return self.saver.save(save_path)

from basic_data import WordConverter, TrainData, TestData, load_description
def train(fn_train, fn_test, fn_word, fn_description, batch_size, num_epoch, embed_size, margin, model_name, load=False, lr=0.001):
    # Initialize data
    wc = WordConverter(fn_word)
    ent2desc = load_description(fn_description, wc)
    train_data = TrainData(fn_train, ent2desc, wc, QUESTION_LEN, DESCRIPTION_LEN)
    test_data = TestData(fn_test, ent2desc, wc, QUESTION_LEN, DESCRIPTION_LEN)
    num_batch = train_data.num_line / batch_size

    run_dir = os.path.join(os.path.abspath(os.path.curdir), 'runs', model_name)
    checkpoint_path = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    save_path = os.path.join(checkpoint_path, 'model')
    log_path = os.path.join(run_dir, 'train.log')
    config_path = os.path.join(run_dir, model_name + '_conf.json')

    model = EntitySim(wc.num_word, embed_size, margin, lr, save_path if load else None)
    best_res = test_data.evaluate(model)
    for i in range(num_epoch):
        loss = 0.
        for j in range(num_batch):
            ques, desc, neg_desc = train_data.next_batch(batch_size)
            loss_one = model.fit(ques, desc, neg_desc)
            loss += loss_one
        old_path = model.save('%s-%s' % (save_path, i))
        res = test_data.evaluate(model)
        if res > best_res:
            best_res = res
            os.rename(old_path, save_path)
            os.rename('%s.meta' % old_path, '%s.meta' % save_path)
            print 'best model', old_path


if __name__ == "__main__":
    pass