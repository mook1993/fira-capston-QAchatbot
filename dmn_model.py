import tensorflow as tf
import numpy as np
import itertools
import pickle
import gensim
from konlpy.tag import Twitter

class dmn_model():

    def __init__(self):
        self.file_glove = "./model/data/glove_wordmap.pickle"
        self.file_output = "./model/data/output.pickle"
        self.file_model = "./model/saver/model.ckpt"
        
        self.num_classes = 3 # 답변 유형 수
        self.twt = Twitter()
        self.sent2tkns = lambda sentence: [ele[0]+'//'+ele[1] for ele in self.twt.pos(sentence)]
        
        self.glove_wordmap = {}
        self.wvecs = []

        self.sess = None
        self.saver = None
        
    def load_glove(self):
        with open(self.file_glove, 'rb') as f:
            self.glove_wordmap=pickle.load(f)
            
        for item in self.glove_wordmap.items():
            self.wvecs.append(item[1])
        
        # Gather the distribution hyperparameters
        self.s = np.vstack(self.wvecs)        
        self.v = np.var(self.s,0)
        self.m = np.mean(self.s,0)
        self.RS = np.random.RandomState(1234)

    # 사전에 없는 단어 처리
    def fill_unk(self, unk):
        self.glove_wordmap[unk] = self.RS.multivariate_normal(self.m, np.diag(self.v))
        
        return self.glove_wordmap[unk]

    # 문장을 토큰화하여주는 함수 (벡터seq, 단어seq)
    def sentence2sequence(self, sentence):
        tokens = self.sent2tkns(sentence)
        rows = []
        words = []
        for word in tokens:
            if word in self.glove_wordmap:
                rows.append(self.glove_wordmap[word])
                words.append(word)
            else:
                rows.append(self.fill_unk(word))
                words.append(word)
                
        return np.array(rows), words
    
    def contextualize_str(self, input_str):
        data = []
        context = []
        for line in input_str.split('\n'):
            l, ine = tuple(line.split(" ", 1))
            
            if l is "1":
                context = []
            if '?' in ine:
                question = ine
                data.append((
                     tuple(zip(*context))+
                     self.sentence2sequence(question)+
                     self.sentence2sequence('None')+
                     ([-1],)+([-1],) ))
            else:
                context.append(self.sentence2sequence(ine[:-1]))
                
        return data
    
    def finalize(self, data):
        final_data = []
        for cqas in data:
            contextvs, contextws, qvs, qws, avs, aws, spt, temp = cqas
            lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
            context_vec = np.concatenate(contextvs)
            context_words=[]
            [context_words.extend(ele) for ele in contextws]

            sentence_ends = np.array(list(lengths))
            final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws, temp)) #!!!!!!!!!!!
        
        return np.array(final_data)
    
    # seq형태의 값을 one hot 인코딩 형태로 변환, ex) [1,2,0] ==> [[0,1,0], [0,0,1], [1,0,0]]
    def seq2onehot(self, seq):
        ret = np.zeros((seq.shape[0], self.num_classes))
        ret[np.arange(seq.shape[0]),seq] = 1
        
        return ret
    
    
    def attention(self, c, mem, existing_facts):
        with tf.variable_scope("attending") as scope:
            attending = tf.concat([c, mem, self.re_q, c * self.re_q,  c * mem, (c-self.re_q)**2, (c-mem)**2], 2)                
            m1 = tf.matmul(attending * existing_facts,
                           tf.tile(self.w_1, tf.stack([tf.shape(attending)[0],1,1]))) * existing_facts                
            bias_1 = self.b_1 * existing_facts
            tnhan = tf.nn.relu(m1 + bias_1)

            m2 = tf.matmul(tnhan, tf.tile(self.w_2, tf.stack([tf.shape(attending)[0],1,1])))

            bias_2 = self.b_2 * existing_facts
            norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

            softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
            softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_idx)
            softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
            softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
                
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)),-1)
    
    def load_graph(self):
        tf.reset_default_graph()
        
        recurrent_cell_size = 128
        D = 100
        learning_rate_dmn = 0.00003
        input_p, output_p = 0.5, 0.5
        batch_size = 3
        passes = 4        
        ff_hidden_size = 256
        weight_decay = 0.00000001
        
        # input context Module
        context = tf.placeholder(tf.float32, [None, None, D], "context")
        self.context_placeholder = context
        
        self.input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

        # recurrent_cell_size: the number of hidden units in recurrent layers.
        input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)
        gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)
        input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope = "input_module")

        # cs: the facts gathered from the context.
        cs = tf.gather_nd(input_module_outputs, self.input_sentence_endings)
        s = input_module_outputs


        # Question Module
        self.query = tf.placeholder(tf.float32, [None, None, D], "query")
        self.input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

        question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, self.query, dtype=tf.float32, scope = tf.VariableScope(True, "input_module"))

        # q: the question states. A [batch_size, recurrent_cell_size] tensor.
        q = tf.gather_nd(question_module_outputs, self.input_query_lengths)
        
        
        # Episodic Memory Moduel      
        size = tf.stack([tf.constant(1),tf.shape(cs)[1], tf.constant(1)])
        self.re_q = tf.tile(tf.reshape(q,[-1,1,recurrent_cell_size]),size)

        output_size = 1

        attend_init = tf.random_normal_initializer(stddev=0.1)
        self.w_1 = tf.get_variable("attend_w1", [1,recurrent_cell_size*7, recurrent_cell_size],
                              tf.float32, initializer = attend_init)
        self.w_2 = tf.get_variable("attend_w2", [1,recurrent_cell_size, output_size],
                              tf.float32, initializer = attend_init)

        self.b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size],
                              tf.float32, initializer = attend_init)
        self.b_2 = tf.get_variable("attend_b2", [1, output_size],
                              tf.float32, initializer = attend_init)
        
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.w_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.b_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.w_2))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.b_2))


        # facts_0s: a [batch_size, max_facts_length, 1] tensor
        #     whose values are 1 if the corresponding fact exists and 0 if not.
        facts_0s = tf.cast(tf.count_nonzero(self.input_sentence_endings[:,:,-1:],-1,keepdims=True),tf.float32)


        with tf.variable_scope("Episodes") as scope:
            attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)
            
            memory = [q]
            attends = []
            
            for a in range(passes):
                attend_to = self.attention(cs, tf.tile(tf.reshape(memory[-1],[-1,1,recurrent_cell_size]),size), facts_0s)
                retain = 1-attend_to
                while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
                update_state = (lambda state, index: (attend_to[:,index,:] *
                                                         attention_gru(cs[:,index,:], state)[0] +
                                                         retain[:,index,:] * state))
                
                memory.append(tuple(tf.while_loop(while_valid_index,
                                  (lambda state, index: (update_state(state,index),index+1)),
                                   loop_vars = [memory[-1], 0]))[0])

                attends.append(attend_to)
                
                scope.reuse_variables()


        # Answer Memory Moduel         
        a0 = tf.concat([memory[-1], q], -1)
        fc_init = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope("answer"):
            
            w_answer1 = tf.get_variable("weight1", [recurrent_cell_size*2, recurrent_cell_size], tf.float32, initializer = fc_init)
            w_answer2 = tf.get_variable("weight2", [recurrent_cell_size, D], tf.float32, initializer = fc_init)
            b_answer1= tf.get_variable("bias1", [recurrent_cell_size], tf.float32, initializer=fc_init)
            b_answer2= tf.get_variable("bias2", [D], tf.float32, initializer=fc_init)

            
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(w_answer1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(w_answer2))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(b_answer1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(b_answer2))

            
            logit = tf.expand_dims(tf.matmul(tf.matmul(a0, w_answer1)+b_answer1, w_answer2)+b_answer2,1)

            
            with tf.variable_scope("ending"):
                all_ends = tf.reshape(self.input_sentence_endings, [-1,2])
                range_ends = tf.range(tf.shape(all_ends)[0])
                ends_indices = tf.stack([all_ends[:,0],range_ends], axis=1)
                ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:,1], [tf.shape(q)[0], tf.shape(all_ends)[0]]), axis=-1)
                range_ind = tf.range(tf.shape(ind)[0])
                mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1),
                                                  tf.ones_like(range_ind), [tf.reduce_max(ind)+1,
                                                                            tf.shape(ind)[0]]), bool)
                
                mask = tf.scan(tf.logical_xor,mask_ends, tf.ones_like(range_ind, dtype=bool))

            
            logits = -tf.reduce_sum(tf.square(context*tf.transpose(tf.expand_dims(
                            tf.cast(mask, tf.float32),-1),[1,0,2]) - logit), axis=-1)


        # 단답생성 Training
        # gold_standard: The real answers.
        self.gold_standard = tf.placeholder(tf.float32, [None, 1, D], "real_answer")
        with tf.variable_scope('accuracy'):
            eq = tf.equal(context, self.gold_standard)
            self.corrbool = tf.reduce_all(eq,-1, name='corrbool')
            logloc = tf.reduce_max(logits, -1, keepdims = True)
            self.locs = tf.equal(logits, logloc, name='locs')

            correctsbool = tf.reduce_any(tf.logical_and(self.locs, self.corrbool), -1)
            corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                                tf.zeros_like(correctsbool,dtype=tf.float32))

            corr = tf.where(self.corrbool, tf.ones_like(self.corrbool, dtype=tf.float32),
                                tf.zeros_like(self.corrbool,dtype=tf.float32))
            
        with tf.variable_scope("loss"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.nn.l2_normalize(logits,-1),
                                                           labels = corr)
            tf.summary.histogram('./loss', loss)  # summary 추가!!!!!!!!!!!!
            total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        optimizer = tf.train.AdamOptimizer(0.0001)
        opt_op = optimizer.minimize(total_loss)



        # 질문과 템플릿 매치되도록 학습
        hidden_size = 256  # output from the LSTM. 5 to directly predict one-hot
        learning_rate_template = 0.001
        qvec_len=128

        # # !!!템플릿 라벨 값은 파일로 부터 파싱하도록 수정
        # y_label=np.array([[1.,0.,0.],
        #          [0.,1.,0.],
        #          [0.,0.,1.]])

        X = tf.reshape(q, [-1, qvec_len])  # X one-hot
        self.Y= tf.placeholder(tf.float32, [None, self.num_classes])

        softmax_W=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[qvec_len, self.num_classes]))
        softmax_b=tf.Variable(tf.zeros([self.num_classes]))
        self.answer_template_pred=tf.nn.softmax(tf.matmul(X,softmax_W)+softmax_b, name='answer_template_pred')
        answer_template_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.answer_template_pred, labels=self.Y))
        answer_template_opt_op=tf.train.AdamOptimizer(learning_rate_template).minimize(answer_template_loss)

    
    def input_prep_batch(self, batch_data, more_data = False, isTest=False):
        context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs, sps, temps = zip(*batch_data)

        temp_labels = self.seq2onehot(np.array(temps))

        ends = list(sentence_ends)
        maxend = max(map(len, ends))
        aends = np.zeros((len(ends), maxend))
        for index, i in enumerate(ends):
            for indexj, x in enumerate(i):
                aends[index, indexj] = x-1
        new_ends = np.zeros(aends.shape+(2,))

        for index, x in np.ndenumerate(aends):
            new_ends[index+(0,)] = index[0]
            new_ends[index+(1,)] = x

        contexts = list(context_vec)
        max_context_length = max([len(x) for x in contexts])
        contextsize = list(np.array(contexts[0]).shape)
        contextsize[0] = max_context_length
        final_contexts = np.zeros([len(contexts)]+contextsize)

        contexts = [np.array(x) for x in contexts]
        for i, context in enumerate(contexts):
            final_contexts[i,0:len(context),:] = context
        max_query_length = max(len(x) for x in questionvs)
        querysize = list(np.array(questionvs[0]).shape)
        querysize[:1] = [len(questionvs),max_query_length]
        queries = np.zeros(querysize)
        querylengths = np.array(list(zip(range(len(questionvs)),[len(q)-1 for q in questionvs])))
        questions = [np.array(q) for q in questionvs]

        for i, question in enumerate(questions):
            queries[i,0:len(question),:] = question

        data = {self.context_placeholder: final_contexts, 
                self.input_sentence_endings: new_ends,
                self.query:queries, 
                self.input_query_lengths:querylengths, 
                self.gold_standard: answervs, 
                self.Y:temp_labels}
        return (data, context_words, cqas) if more_data else data
    
    
    def makeFormatted(self, input_str):
        input_contextual_data = self.contextualize_str(input_str)
        input_final_data = self.finalize(input_contextual_data)
        input_set, input_context_words, input_cqas = self.input_prep_batch(input_final_data, more_data=True, isTest=True)
        return [input_set, input_context_words, input_cqas]

    def getPretrainedModel(self):

        if self.saver is None:
            self.saver = tf.train.Saver()

            _init = tf.global_variables_initializer()
        
            self.sess = tf.Session()
            self.sess.run(_init)
            
            self.saver.restore(self.sess, self.file_model)        
        
        return self.sess

    # 올바른 답을 주기위해서 미리학습한 모델의 session을 인자로 줌
    def getAnswerInfo(self, input_set=[]):
        output_ancr = self.sess.run([self.corrbool, self.locs], feed_dict=input_set)
        output_template= self.sess.run([self.answer_template_pred], feed_dict=input_set)
        output_a = output_ancr[0]
        output_n = output_ancr[1]
        
        with open(self.file_output, 'wb') as f:
            pickle.dump([output_a, output_n, output_template], f)
            
        return [output_a, output_n, output_template]
    
    def getAnswer(self, old_input_str, new_input_str):

        answers=[]
        a_template_dict = {0: '%s을 받을 수 있습니다.',
                   1: '%s인 경우 받을 수 없습니다.',
                   2: '보험계약일로부터 %s이 되는 시점의 계약해당일의 전일까지입니다.'}
                

        with tf.Session() as sess:
            self.sess = sess       
            self.sess.run(tf.global_variables_initializer())
            
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.file_model)           
                
            input_set, input_context_words, input_cqas = self.makeFormatted(old_input_str)
            _, new_input_context_words_, __ = self.makeFormatted(new_input_str)

            output_a, output_n, output_template = self.getAnswerInfo(input_set=input_set)

            # Locations of responses within contexts
            output_indices = np.argmax(output_n,axis=1)

            # Locations of actual answers within contexts
            output_indicesc = np.argmax(output_a,axis=1)
            
            # print(val_context_words)
            output_answer_templates=np.argmax(output_template[0], axis=1)
            
            for i_,e_,cw_, cqa_, a_tmp_ in list(zip(output_indices, output_indicesc, new_input_context_words_, input_cqas, output_answer_templates)):
                answers.append(a_template_dict[a_tmp_]%(cw_[i_].split('//')[0]))           
        
            
        return answers
