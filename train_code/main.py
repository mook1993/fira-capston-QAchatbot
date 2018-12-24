# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib
import sys
import os
import zipfile
import tarfile
import json 
import hashlib
import re
import itertools
import pickle
import gensim
from konlpy.tag import Twitter


# In[15]:

model=gensim.models.Word2Vec.load('../dataset/glove/insurance_100D.model')


# In[28]:


num_classes=3 # 답변 유형 수


# In[72]:


twt=Twitter()

sent2tkns=lambda sentence: [ele[0]+'//'+ele[1] for ele in twt.pos(sentence)]

# Deserialize GloVe vectors
glove_wordmap = {}
for ele in model.wv.vocab.items():
    glove_wordmap[ele[0]]=model[ele[0]]
        
wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

# Gather the distribution hyperparameters
v = np.var(s,0) 
m = np.mean(s,0) 
RS = np.random.RandomState()

# 사전에 없는 단어 처리
def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(m,np.diag(v))
    return glove_wordmap[unk]

# 문장을 토큰화하여주는 함수 (벡터seq, 단어seq)
def sentence2sequence(sentence):
    tokens = sent2tkns(sentence)
    rows = []
    words = []
    for word in tokens:
        if word in glove_wordmap:
            rows.append(glove_wordmap[word])
            words.append(word)
        else:
            rows.append(fill_unk(word))
            words.append(word)
    return np.array(rows), words

def contextualize_str(input_str):
    """
    0: context의 벡터
    1: context의 단어토큰
    2: question의 벡터
    3: question의 단어토큰
    4: question의 벡터
    5: answer의 벡터
    6: answer의 단어
    7: supporting factor
    8: template index
    """
    data = []
    context = []
    for line in input_str.split('\n'):
        l, ine = tuple(line.split(" ", 1))
        # Split the line numbers from the sentences they refer to.
        if l is "1":
            # New contexts always start with 1, 
            # so this is a signal to reset the context.
            context = []
        if '?' in ine:
            # Tabs are the separator between questions and answers,
            # and are not present in context statements.
            question = ine
            print(question)
            data.append((
                tuple(zip(*context))+
                 sentence2sequence(question)+
                 sentence2sequence('None')+
                 ([-1],)+
                 tuple([-1])
                ))
            # Multiple questions may refer to the same context, so we don't reset it.
        else:
            # Context sentence.
            context.append(sentence2sequence(ine[:-1]))
    return data

def finalize(data):
    """
        0: context의 단어 벡터
        1: 문장마지막 인덱스
        2: question 벡터
        3: supporting factor
        4: context의 단어 토큰들
        5: cqas
        6: answer 벡터
        7: answer 단어
        8: template 정보
    """
    final_data = []
    for cqas in data:
        contextvs, contextws, qvs, qws, avs, aws, spt, temp = cqas
        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words=[]
        [context_words.extend(ele) for ele in contextws]

        # Location markers for the beginnings of new sentences.
        sentence_ends = np.array(list(lengths)) 
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws, temp)) #!!!!!!!!!!!
    return np.array(final_data)

# seq형태의 값을 one hot 인코딩 형태로 변환, ex) [1,2,0] ==> [[0,1,0], [0,0,1], [1,0,0]]
def seq2onehot(seq):
    ret=np.zeros((seq.shape[0],num_classes))
    ret[np.arange(seq.shape[0]),seq]=1
    return ret


def input_prep_batch(batch_data, more_data = False, isTest=False):        
    context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs, sps, temps = zip(*batch_data)
    
    temp_labels=seq2onehot(np.array(temps))
    
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
    
    data = {context_placeholder: final_contexts, input_sentence_endings: new_ends, 
                            query:queries, input_query_lengths:querylengths, gold_standard: answervs, Y:temp_labels}
    return (data, context_words, cqas) if more_data else data



tf.reset_default_graph()
# Hyperparameters

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 128

# The number of dimensions in our word vectorizations.
D = 100

# How quickly the network learns. Too high, and we may run into numeric instability 
# or other issues.
learning_rate_dmn = 0.00003

# Dropout probabilities. For a description of dropout and what these probabilities are, 
# see Entailment with TensorFlow.
input_p, output_p = 0.5, 0.5

# How many questions we train on at a time.
batch_size = 3

# Number of passes in episodic memory. We'll get to this later.
passes = 4

# Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
ff_hidden_size = 256

weight_decay = 0.00000001


# In[75]:


# Input Module

# Context: A [batch_size, maximum_context_length, word_vectorization_dimensions] tensor 
# that contains all the context information.
context = tf.placeholder(tf.float32, [None, None, D], "context")  
context_placeholder = context # I use context as a variable name later on

# input_sentence_endings: A [batch_size, maximum_sentence_count, 2] tensor that 
# contains the locations of the ends of sentences. 
input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

# recurrent_cell_size: the number of hidden units in recurrent layers.
input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

# input_p: The probability of maintaining a specific hidden input unit.
# Likewise, output_p is the probability of maintaining a specific hidden output unit.
gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)

# dynamic_rnn also returns the final internal state. We don't need that, and can
# ignore the corresponding output (_). 
input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope = "input_module")

# cs: the facts gathered from the context.
cs = tf.gather_nd(input_module_outputs, input_sentence_endings)
# to use every word as a fact, useful for tasks with one-sentence contexts
s = input_module_outputs


# In[76]:


# Question Module

# query: A [batch_size, maximum_question_length, word_vectorization_dimensions] tensor 
#  that contains all of the questions.

query = tf.placeholder(tf.float32, [None, None, D], "query")

# input_query_lengths: A [batch_size, 2] tensor that contains question length information. 
# input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range() 
# so that it plays nice with gather_nd.
input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32, 
                                               scope = tf.VariableScope(True, "input_module"))

# q: the question states. A [batch_size, recurrent_cell_size] tensor.
q = tf.gather_nd(question_module_outputs, input_query_lengths)


# In[77]:


# Episodic Memory

# make sure the current memory (i.e. the question vector) is broadcasted along the facts dimension
size = tf.stack([tf.constant(1),tf.shape(cs)[1], tf.constant(1)])
re_q = tf.tile(tf.reshape(q,[-1,1,recurrent_cell_size]),size)


# Final output for attention, needs to be 1 in order to create a mask
output_size = 1 

# Weights and biases
attend_init = tf.random_normal_initializer(stddev=0.1)
w_1 = tf.get_variable("attend_w1", [1,recurrent_cell_size*7, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
w_2 = tf.get_variable("attend_w2", [1,recurrent_cell_size, output_size], 
                      tf.float32, initializer = attend_init)

b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
b_2 = tf.get_variable("attend_b2", [1, output_size], 
                      tf.float32, initializer = attend_init)

# Regulate all the weights and biases
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))


def attention(c, mem, existing_facts):
    """
    Custom attention mechanism.
    c: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor 
        that contains all the facts from the contexts.
    mem: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that 
        contains the current memory. It should be the same memory for all facts for accurate results.
    existing_facts: A [batch_size, maximum_sentence_count, 1] tensor that 
        acts as a binary mask for which facts exist and which do not.

    """
    with tf.variable_scope("attending") as scope:
        # attending: The metrics by which we decide what to attend to.
        attending = tf.concat([c, mem, re_q, c * re_q,  c * mem, (c-re_q)**2, (c-mem)**2], 2)

        # m1: First layer of multiplied weights for the feed-forward network. 
        #     We tile the weights in order to manually broadcast, since tf.matmul does not
        #     automatically broadcast batch matrix multiplication as of TensorFlow 1.2.
        m1 = tf.matmul(attending * existing_facts, 
                       tf.tile(w_1, tf.stack([tf.shape(attending)[0],1,1]))) * existing_facts
        # bias_1: A masked version of the first feed-forward layer's bias
        #     over only existing facts.

        bias_1 = b_1 * existing_facts

        # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity; 
        #        choosing relu was a design choice intended to avoid issues with 
        #        low gradient magnitude when the tanh returned values close to 1 or -1. 
        tnhan = tf.nn.relu(m1 + bias_1)

        # m2: Second layer of multiplied weights for the feed-forward network. 
        #     Still tiling weights for the same reason described in m1's comments.
        m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0],1,1])))

        # bias_2: A masked version of the second feed-forward layer's bias.
        bias_2 = b_2 * existing_facts

        # norm_m2: A normalized version of the second layer of weights, which is used 
        #     to help make sure the softmax nonlinearity doesn't saturate.
        norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

        # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor. 
        #     We make norm_m2 a sparse tensor, then make it dense again after the operation.
        softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
        softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_idx)
        softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
        softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)),-1)

# facts_0s: a [batch_size, max_facts_length, 1] tensor 
#     whose values are 1 if the corresponding fact exists and 0 if not.
facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:,:,-1:],-1,keepdims=True),tf.float32)


with tf.variable_scope("Episodes") as scope:
    attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

    # memory: A list of all tensors that are the (current or past) memory state 
    #   of the attention mechanism.
    memory = [q]

    # attends: A list of all tensors that represent what the network attends to.
    attends = []
    for a in range(passes):
        # attention mask
        attend_to = attention(cs, tf.tile(tf.reshape(memory[-1],[-1,1,recurrent_cell_size]),size), facts_0s)

        # Inverse attention mask, for what's retained in the state.
        retain = 1-attend_to

        # GRU pass over the facts, according to the attention mask.
        while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
        update_state = (lambda state, index: (attend_to[:,index,:] * 
                                                 attention_gru(cs[:,index,:], state)[0] + 
                                                 retain[:,index,:] * state))
        # start loop with most recent memory and at the first index
        memory.append(tuple(tf.while_loop(while_valid_index,
                          (lambda state, index: (update_state(state,index),index+1)),
                           loop_vars = [memory[-1], 0]))[0]) 

        attends.append(attend_to)

        # Reuse variables so the GRU pass uses the same variables every pass.
        scope.reuse_variables()


# In[78]:


# Answer Module

# a0: Final memory state. (Input to answer module)
a0 = tf.concat([memory[-1], q], -1)

# fc_init: Initializer for the final fully connected layer's weights.
fc_init = tf.random_normal_initializer(stddev=0.1) 

with tf.variable_scope("answer"):
    # w_answer: The final fully connected layer's weights.
    w_answer1 = tf.get_variable("weight1", [recurrent_cell_size*2, recurrent_cell_size], tf.float32, initializer = fc_init)
    w_answer2 = tf.get_variable("weight2", [recurrent_cell_size, D], tf.float32, initializer = fc_init)
    b_answer1= tf.get_variable("bias1", [recurrent_cell_size], tf.float32, initializer=fc_init)
    b_answer2= tf.get_variable("bias2", [D], tf.float32, initializer=fc_init)

    # Regulate the fully connected layer's weights
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(w_answer1)) 
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(w_answer2)) 
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(b_answer1)) 
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(b_answer2)) 
    
    # The regressed word. This isn't an actual word yet;
    #    we still have to find the closest match.
    logit = tf.expand_dims(tf.matmul(tf.matmul(a0, w_answer1)+b_answer1, w_answer2)+b_answer2,1)

    # Make a mask over which words exist.
    with tf.variable_scope("ending"):
        all_ends = tf.reshape(input_sentence_endings, [-1,2])
        range_ends = tf.range(tf.shape(all_ends)[0])
        ends_indices = tf.stack([all_ends[:,0],range_ends], axis=1)
        ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:,1], [tf.shape(q)[0], tf.shape(all_ends)[0]]), axis=-1)
        range_ind = tf.range(tf.shape(ind)[0])
        mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1), 
                                          tf.ones_like(range_ind), [tf.reduce_max(ind)+1, 
                                                                    tf.shape(ind)[0]]), bool)
        # A bit of a trick. With the locations of the ends of the mask (the last periods in 
        #  each of the contexts) as 1 and the rest as 0, we can scan with exclusive or 
        #  (starting from all 1). For each context in the batch, this will result in 1s 
        #  up until the marker (the location of that last period) and 0s afterwards.
        mask = tf.scan(tf.logical_xor,mask_ends, tf.ones_like(range_ind, dtype=bool))

    # We score each possible word inversely with their Euclidean distance to the regressed word.
    #  The highest score (lowest distance) will correspond to the selected word.
    logits = -tf.reduce_sum(tf.square(context*tf.transpose(tf.expand_dims(
                    tf.cast(mask, tf.float32),-1),[1,0,2]) - logit), axis=-1)


# In[79]:


# 단답생성 Training

# gold_standard: The real answers.
gold_standard = tf.placeholder(tf.float32, [None, 1, D], "real_answer")
with tf.variable_scope('accuracy'):
    eq = tf.equal(context, gold_standard)
    corrbool = tf.reduce_all(eq,-1)
    logloc = tf.reduce_max(logits, -1, keepdims = True)
    # locs: A boolean tensor that indicates where the score
    #  matches the minimum score. This happens on multiple dimensions, 
    #  so in the off chance there's one or two indexes that match 
    #  we make sure it matches in all indexes.
    locs = tf.equal(logits, logloc)

    # correctsbool: A boolean tensor that indicates for which 
    #   words in the context the score always matches the minimum score.
    correctsbool = tf.reduce_any(tf.logical_and(locs, corrbool), -1)
    # corrects: A tensor that is simply correctsbool cast to floats.
    corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                        tf.zeros_like(correctsbool,dtype=tf.float32))

    # corr: corrects, but for the right answer instead of our selected answer.
    corr = tf.where(corrbool, tf.ones_like(corrbool, dtype=tf.float32), 
                        tf.zeros_like(corrbool,dtype=tf.float32))
with tf.variable_scope("loss"):
    # Use sigmoid cross entropy as the base loss, 
    #  with our distances as the relative probabilities. There are
    #  multiple correct labels, for each location of the answer word within the context.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.nn.l2_normalize(logits,-1),
                                                   labels = corr)
    tf.summary.histogram('./loss', loss)  # summary 추가!!!!!!!!!!!!
    # Add regularization losses, weighted by weight_decay.
    total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# TensorFlow's default implementation of the Adam optimizer works. We can adjust more than 
#  just the learning rate, but it's not necessary to find a very good optimum.
optimizer = tf.train.AdamOptimizer(0.0001)

# Once we have an optimizer, we ask it to minimize the loss 
#   in order to work towards the proper training.
opt_op = optimizer.minimize(total_loss)


# In[80]:


# 질문과 템플릿 매치되도록 학습
hidden_size = 256  # output from the LSTM. 5 to directly predict one-hot
learning_rate_template = 0.001
qvec_len=128
num_classes=3 # 답변 유형 수

# # !!!템플릿 라벨 값은 파일로 부터 파싱하도록 수정
# y_label=np.array([[1.,0.,0.],
#          [0.,1.,0.],
#          [0.,0.,1.]])

X = tf.reshape(q, [-1, qvec_len])  # X one-hot
Y= tf.placeholder(tf.float32, [None, num_classes])

softmax_W=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[qvec_len, num_classes]))
softmax_b=tf.Variable(tf.zeros([num_classes]))
answer_template_pred=tf.nn.softmax(tf.matmul(X,softmax_W)+softmax_b)
answer_template_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=answer_template_pred, labels=Y))
answer_template_opt_op=tf.train.AdamOptimizer(learning_rate_template).minimize(answer_template_loss)


# In[81]:


# Initialize variables
init = tf.global_variables_initializer()

# Launch the TensorFlow session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)

# create tenseorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./tensorboard/train', sess.graph)
test_writer = tf.summary.FileWriter('./tensorboard/test')


# In[82]:


saver=tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver.restore(sess, "./tmp/model.ckpt")


# In[83]:


input_str='''1 지급액은 1000만원 - 이미 지급된 건강진단보험금입니다.
2 유족 위로금에 관한 항목입니다. 
3 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다. 
4 지급액은 300만원입니다.
5 보험기간(종신) 중 피보험자가 사망하였을 때:
6 사망보험금 및 유족위로금
7 사망 보험금에 관한 항목입니다
8 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다.
9 피보험자인 부모님이 사망하셨는데 받을 수 있는 사망보험금이 얼마나 되나요?
10 가족이 사망하게될 시에 지급액은 얼마가 나오나요?
12 자신이 죽게되면 지급액은 얼마인가요?
14 자신이 죽게되면 지급액은 어떻게 되나요?
16 본인이 사망할 시에 지급액은 얼마가 나오나요?
18 본인이 죽게될 시에 지급액은 어떻게 되나요?
20 가족이 죽게되면 지급액은 어떻게 되나요?'''

a_template_dict={0:'%s을 받을 수 있습니다.', 
                 1:'%s인 경우 받을 수 없습니다.',
                 2:'보험계약일로부터 %s이 되는 시점의 계약해당일의 전일까지입니다.'}


input_contextual_data = contextualize_str(input_str)
input_final_data = finalize(input_contextual_data)
input_set, input_context_words, input_cqas = input_prep_batch(input_final_data, more_data=True, isTest=True)

def getAnswerInfo(stand_state=False):
    if stand_state:
        with open('../dataset/variable/output.pickle', 'rb') as f:
            data=pickle.load(f)
            return data
    else:
        output_ancr = sess.run([corrbool, locs], feed_dict=input_set)
        output_template=sess.run([answer_template_pred], feed_dict=input_set)
        output_a = output_ancr[0]
        output_n = output_ancr[1]
        with open('../dataset/variable/output.pickle', 'wb') as f:
            pickle.dump([output_a, output_n, output_template], f)
        return [output_a, output_n, output_template]


def getAnswer(input_str):
    output_a, output_n, output_template = getAnswerInfo(True)

    # Locations of responses within contexts
    output_indices = np.argmax(output_n,axis=1)

    # Locations of actual answers within contexts
    output_indicesc = np.argmax(output_a,axis=1)
    # print(val_context_words)
    output_answer_templates=np.argmax(output_template[0], axis=1)

    answers=[]
    for i_,e_,cw_, cqa_, a_tmp_ in list(zip(output_indices, output_indicesc, input_context_words, input_cqas, output_answer_templates)):
    #     ccc_ = " ".join(cw_)
    #     print("TEXT: ",ccc)
    #     if(cw_[e_]==cw_[i_]):
    #     print ("QUESTION: ", " ".join(cqa_[3]))
    #     print("EXPECTED: ", cw_[e_])
    #     print ("RESPONSE: ", cw_[i_], ["Correct", "Incorrect"][i_!=e_])
    #     print("Sentence Response:",a_template_dict[a_tmp_]%(cw_[i_].split('//')[0]))
    #     print()
        answers.append(a_template_dict[a_tmp_]%(cw_[i_].split('//')[0]))
    return answers

print(getAnswer(input_str))


