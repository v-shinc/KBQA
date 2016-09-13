import numpy
import tensorflow as tf
import os
import json
import sys




SENTENCE_LEN = 10
MEMORY_SIZE = 5  # number of sentences in one description
class ContextMatch(object):
    def __init__(self, num_word, embed_size, memory_size, sentence_len, lr, load_path):
        '''q: [batch_size, embed_size]
           sentences: [batch_size, nsentence, embed_size]
        '''
        with tf.device('/gpu:0'):
            self.questions = tf.placeholder(tf.int32, [None, sentence_len], name='questions')
            self.contexts = tf.placeholder(tf.int32, [None, memory_size, sentence_len], name='descriptions')
            self.label = tf.placeholder(tf.float32, [None], name='labels')
            initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
            A = tf.get_variable('A', [num_word, embed_size], initializer=initializer)
            C = tf.get_variable('C', [num_word, embed_size], initializer=initializer)

            # Temporal Encoding
            # T_A = tf.get_variable('T_A', [MEMORY_SIZE, embed_size], initializer=initializer)
            # T_C = tf.get_variable('T_C', [MEMORY_SIZE, embed_size], initializer=initializer)

            # m_i = sum A_ij * x_ij + T_A_i

            context_memory = tf.reduce_sum(tf.nn.embedding_lookup(A, self.contexts), 2)
            description_sum = tf.reduce_sum(tf.nn.embedding_lookup(C, self.contexts), 2)
            question_sum = tf.reduce_sum(tf.nn.embedding_lookup(C, self.questions), 1)

            question_3d_sum = tf.reshape(question_sum, [-1, 1, embed_size])
            # tf.batch_matmul(question_3d_sum, description_memory, adj_y=True)
            dot = tf.reduce_sum(tf.mul(question_3d_sum, context_memory), -1)
            p = tf.nn.softmax(tf.reshape(dot, [-1, memory_size]))

            p_3d = tf.reshape(p, [-1, memory_size, 1])
            weighted_context = tf.reduce_sum(tf.mul(p_3d, description_sum), 1)
            self.logits = tf.reduce_sum(tf.mul(question_sum, weighted_context), -1)
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.label))

            opt = tf.train.AdamOptimizer(lr)
            self.train_op = opt.minimize(self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.session = tf.InteractiveSession(config=config)
            self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
            if load_path:
                self.saver.restore(self.session, load_path)
            else:
                self.session.run(tf.initialize_all_variables())

    def predict(self, questions, context):
        scores = self.session.run(
            self.logits,
            {self.questions: questions,
             self.contexts: context}
        )
        return scores

    def init_from_conf(self, filename):
        conf = json.load(filename, encoding='utf8')
        load_path = os.path.abspath(os.path.join(os.path.curdir, 'runs', conf['model_name'], 'checkpoints', 'model'))
        return ContextMatch(conf['num_word'], conf['embed_size'], conf['memory_size'], conf['sentence_len'], conf['lr'], load_path)

    def fit(self, questions, contexts, labels):
        _, loss = self.session.run([self.train_op, self.loss],{
            self.questions: questions,
            self.contexts: contexts,
            self.label: labels
        })
        return loss

    def save(self, save_path):
        return self.saver.save(self.session, save_path)

from mn_data import load_description, MNTrainData, MNTestData, WordConverter
def train(fn_train, fn_test, fn_word, fn_description, batch_size, num_epoch, embed_size, model_name, load=False, lr=0.001):
    params = locals()
    print '[In train] init word converter'
    wc = WordConverter(fn_word)
    print '[In train] load description'
    ent2desc = load_description(fn_description, wc)
    print '[In train] init train data'
    train_data = MNTrainData(fn_train, ent2desc, wc, SENTENCE_LEN, MEMORY_SIZE)
    print '[In train] load', train_data.num_line,
    print '[In train] init test data'
    test_data = MNTestData(fn_test, ent2desc, wc, SENTENCE_LEN, MEMORY_SIZE)
    num_batch = train_data.num_line / batch_size

    run_dir = os.path.join(os.path.abspath(os.path.curdir), 'runs', model_name)
    checkpoint_path = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    save_path = os.path.join(checkpoint_path, 'model')
    log_path = os.path.join(run_dir, 'train.log')
    config_path = os.path.join(run_dir, model_name + '_conf.json')
    json.dump(params, open(config_path, 'w'), ensure_ascii=False)

    model = ContextMatch(wc.num_word, embed_size, MEMORY_SIZE, SENTENCE_LEN, lr, save_path if load else None)
    best_res = test_data.evaluate(model, verbose=True)
    try:
        for i in range(num_epoch):
            loss = 0.
            processed = 0
            for j in range(num_batch):
                if j % 10 == 0:
                    sys.stdout.write('process to %d\r' % processed)
                    sys.stdout.flush()


                ques, desc, labels = train_data.next_batch(batch_size)

                processed += len(labels)
                loss_one = model.fit(ques, desc, labels)
                loss += loss_one
            print '#', i, 'loss =', loss
            old_path = model.save('%s-%s' % (save_path, i))
            res = test_data.evaluate(model)
            if res > best_res:
                best_res = res
                os.rename(old_path, save_path)
                os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                print 'best model', old_path
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    fn_train = '../data/entity.train'
    fn_test = '../data/wq.entity.test'
    fn_word = '../data/description.word'
    fn_description = '../data/description.sentences.large.clean'
    train(fn_train, fn_test, fn_word, fn_description, batch_size=100, num_epoch=100, embed_size=60, model_name='wq.memory.description', load=False, lr=0.001)

