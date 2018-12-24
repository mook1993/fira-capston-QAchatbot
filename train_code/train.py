
# coding: utf-8

# In[4]:

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


# In[5]:


datasetDir="../dataset/"
glove_vectors_file =datasetDir+"glove/word2vec/word2vec_model/embedding_100d_final.model"

data_set_zip = datasetDir+"tasks_1-20_v1-2.tar.gz"

train_set_file = "qa_train_insurance.txt"
test_set_file="qa_test_insurance.txt"

train_set_post_file = datasetDir + train_set_file
test_set_post_file=datasetDir+test_set_file


# In[6]:


from konlpy.tag import Twitter
import codecs

twt=Twitter()
sent2tkns=lambda sentence: [ele[0]+'//'+ele[1] for ele in twt.pos(sentence)]
def getContext(filepath):
    with codecs.open(filepath, encoding="utf-8", mode='rb') as file:
        contexts=re.split('[\n]',file.read())
        clean_context=[sent2tkns(sent.strip()[1:].split('\t')[0]) for sent in contexts]
        file.close()
        return clean_context
    
newSentences=getContext(train_set_post_file)[:-1]


# ## pre-trained w2v 모델에 새로운 단어학습

# In[7]:


model = gensim.models.Word2Vec.load(glove_vectors_file)
model.build_vocab(newSentences, update=True, min_count=1)
model.train(newSentences, total_examples=len(newSentences), epochs=300)
model.save('../dataset/glove/insurance_100D.model')


# In[8]:


# model=gensim.models.Word2Vec.load('../dataset/glove/insurance_100D.model')


# In[9]:


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


# In[10]:


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


# ## 1차 데이터셋 생성 cqas

# In[11]:


def contextualize(set_file):
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
    with open(set_file, "r", encoding="utf8") as train:
        for line in train:
            l, ine = tuple(line.split(" ", 1))
            # Split the line numbers from the sentences they refer to.
            if l is "1":
                # New contexts always start with 1, 
                # so this is a signal to reset the context.
                context = []
            if "\t" in ine: 
                # Tabs are the separator between questions and answers,
                # and are not present in context statements.
                question, answer, support, template = tuple(ine.split("\t"))
                data.append((
                    tuple(zip(*context))+
                     sentence2sequence(question)+
                     sentence2sequence(answer)+
                     ([int(s) for s in support.split()],)+
                     tuple([int(template[:-1])])
                    ))
                # Multiple questions may refer to the same context, so we don't reset it.
            else:
                # Context sentence.
                context.append(sentence2sequence(ine[:-1]))
    return data

# 매 학습때마다 train set에서 랜덤하게 추출하여 학습, validation도 마찬가지로 (원래는 test set으로 해야하지만...)
import random

train_data = contextualize(train_set_post_file)
test_data=contextualize(test_set_post_file)
random.shuffle(train_data)
random.shuffle(test_data)
# test_data=dataset[int(len(dataset)*(1-test_ratio)):]


# #### [[단어1벡터 ...], [문장단위인덱스정보],[질문벡터],[supporting fact],[단어1, ..., 마지막단어],[train_data],[답변벡터],[답변word], 템플릿정보]

# In[12]:


final_train_data = []
final_test_data=[]
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

final_train_data = finalize(train_data)   
final_test_data = finalize(test_data)



# ## 단답 생성 네트워크
# #### : DMN을 이용한 질문에 따른 단답생성 메커니즘을 활용

# In[25]:


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
# The strength of our regularization. Increase to encourage sparsity in episodic memory, 
# but makes training slower. Don't make this larger than leraning_rate.

training_iterations_count = 400000
# How many questions the network trains on each time it is trained. 
# Some questions are counted multiple times.

display_step = 100
# How many iterations of training occur before each validation check.


# In[26]:


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


# In[27]:


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


# In[28]:


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


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


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


# In[29]:


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
        ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:,1],
                                          [tf.shape(q)[0], tf.shape(all_ends)[0]]),
                            axis=-1)
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


# In[30]:


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


# ## 질문/답변템플릿 매칭 네트워크
# #### - 동일의미의 다양한 형태의 질문을 한가지 유형의 답변을 묶어줄 수 있도록 네트워크 설계
# #### - question의 last hidden state 값을 input으로 하고 one-hot encoding을 output으로 내게끔 함 

# In[31]:


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


# In[32]:


# Initialize variables
init = tf.global_variables_initializer()

# Launch the TensorFlow session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)

# create tenseorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./tensorboard/train', sess.graph)
test_writer = tf.summary.FileWriter('./tensorboard/test')


# In[33]:


# seq형태의 값을 one hot 인코딩 형태로 변환, ex) [1,2,0] ==> [[0,1,0], [0,0,1], [1,0,0]]
def seq2onehot(seq):
    ret=np.zeros((seq.shape[0],num_classes))
    ret[np.arange(seq.shape[0]),seq]=1
    return ret


# In[34]:


def prep_batch(batch, more_data = False, isTest=False):
    batch_data=[]
    if isTest:
        batch_data=final_test_data[batch]
    else:
        batch_data=final_train_data[batch]
        
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


# #### [[단어1벡터 ...], [문장단위인덱스정보],[질문벡터],[supporting fact],[1단어,...,마지막단어],[train_data],[답변벡터],[답변word]]

# In[24]:


# Use TQDM if installed
tqdm_installed = False
try:
    from tqdm import tqdm
    tqdm_installed = True
except:
    pass


saver=tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver.restore(sess, "./tmp/model.ckpt")

# Prepare validation set
val_batch = np.random.randint(final_test_data.shape[0], size=final_test_data.shape[0])
validation_set, val_context_words, val_cqas = prep_batch(val_batch, more_data=True, isTest=True)

# 질문/답변템플릿 네트워크 학습
def template_train(iterations, batch_size):
    training_iterations = range(0,iterations,batch_size)

    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    wordz = []
    for j in training_iterations:
        batch = np.random.randint(final_train_data.shape[0], size=final_train_data.shape[0])
        sess.run([answer_template_opt_op], feed_dict=prep_batch(batch))
        if j%100==0:        
            pa,ra=sess.run([answer_template_pred, Y], feed_dict=prep_batch(batch))
#             print(pa)
#             print(ra)
            
template_train(100,batch_size) # Small amount of training for preliminary results


# 단답생성 네트워크 학습
def dmn_train(iterations, batch_size):
    training_iterations = range(0,iterations,batch_size)

    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    wordz = []
    for j in training_iterations:
        batch = np.random.randint(final_train_data.shape[0], size=final_train_data.shape[0])
        summ, _ = sess.run([merged, opt_op], feed_dict=prep_batch(batch))
        test_writer.add_summary(summ, j)
        
        if (j) % 10 == 0:
            print(batch)
            print(val_batch)
            acc, ccs, tmp_loss, log, con, cor, loc, a, n  = sess.run([corrects, cs, total_loss, logit, context_placeholder, corr, locs, corrbool, locs], feed_dict=validation_set)
            print("Iter " + str(j/batch_size) + ", Minibatch Loss= ",tmp_loss, "Accuracy= ", np.mean(acc))
        if (j) % 50 ==0:
            saver=tf.train.Saver()
            save_path=saver.save(sess, './tmp/model.ckpt')
            print("Model saved in file: %s" % save_path)
#             indicesc = np.argmax(a,axis=1) # context안에서 정답단어 위치
#             indices = np.argmax(n,axis=1)  # context안에서 예측단어 위치
#             for i in range(len(val_batch)):
#                 real_a=val_context_words[i][indicesc[i]]
#                 pred_a=val_context_words[i][indices[i]]
#                 if real_a!=pred_a:
#                     print(val_batch[i])
#                     print(val_context_words[i])
#                     print(val_cqas[i][3])
#                     print(val_cqas[i][5])
#                     print("정답:",real_a)
#                     print("예측:",pred_a)
#                     print('=========================')
            print()
dmn_train(1000,batch_size) # Small amount of training for preliminary results


# In[25]:


saver=tf.train.Saver()
save_path=saver.save(sess, './tmp/model.ckpt')
print("Model saved in file: %s" % save_path)



# # data
#         0. The sentences in the context in vectorized form.
#         1. The sentences in the context as a list of string tokens.
#         2. The question in vectorized form.
#         3. The question as a list of string tokens.
#         4. The answer in vectorized form.
#         5. The answer as a list of string tokens.
#         6. A list of numbers for supporting statements, which is currently unused.
#         
# # Validation data
# - validation_set: final_data  (context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws)
# - val_context_words: val_context_words
# - val_cqas: val_cqas

# In[27]:


# # attention을 잘하는지 찍어보는 부분
# ancr = sess.run([corrbool, locs, total_loss, logits, facts_0s, w_1] + attends + [query, cs, question_module_outputs], feed_dict=validation_set)
# template=sess.run([answer_template_pred], feed_dict=validation_set)

# a = ancr[0]
# n = ancr[1]
# cr = ancr[2]
# attenders = np.array(ancr[6:-3])
# faq = np.sum(ancr[4], axis=(-1, -2))  # Number of facts in each context
# limit = 5

# Important_sent_idx=np.argmax(attenders, -2)

# import matplotlib.pyplot as plt
# for question in range(min(limit, batch_size)):
#     print(val_cqas[question][3], val_cqas[question][5], val_cqas[question][6])
#     print(val_context_words[question])
    
#     for hop in range(Important_sent_idx.shape[0]):
#             print(str(hop+1)+'hop:', Important_sent_idx[hop][question][0]+1)
            
#     plt.yticks(range(passes, 0, -1))
#     plt.ylabel("Episode")
#     plt.xlabel("Question " + str(question + 1))
#     pltdata = attenders[:, question, :int(faq[question]), 0]

#     # Display only information about facts that actually exist, all others are 0
#     pltdata = (pltdata - pltdata.mean()) / ((pltdata.max() - pltdata.min() + 0.001)) * 256
#     plt.pcolor(pltdata, cmap=plt.cm.BuGn, alpha=0.7)
#     plt.show()


# In[28]:


# print(np.argmax(template[0], axis=1))


# ## template
# #### 0: %s만원 받을 수 있습니다. (지급)
# #### 1: %s인 경우 받을 수 없습니다. (지급 부정)
# #### 2: 보험계약일로부터 %s년이 되는 시점의 계약해당일의 전일까지입니다. (유효기간)

# In[40]:


# # Locations of responses within contexts
# indices = np.argmax(n,axis=1)

# # Locations of actual answers within contexts 
# indicesc = np.argmax(a,axis=1)
# # print(val_context_words)
# answer_templates=np.argmax(template[0], axis=1)
# a_template_dict={0:'%s을 받을 수 있습니다.', 
#                  1:'%s인 경우 받을 수 없습니다.',
#                  2:'보험계약일로부터 %s이 되는 시점의 계약해당일의 전일까지입니다.'}

# for i,e,cw, cqa, a_tmp in list(zip(indices, indicesc, val_context_words, val_cqas, answer_templates)):
#     ccc = " ".join(cw)
# #     print("TEXT: ",ccc)
#     if(cw[e]==cw[i]):
#         print ("QUESTION: ", " ".join(cqa[3]))
#         print("EXPECTED: ", cw[e])
#         print ("RESPONSE: ", cw[i], ["Correct", "Incorrect"][i!=e])
#         print("Sentence Response:",a_template_dict[a_tmp]%(cw[i].split('//')[0]))
#         print()


# In[ ]:


dmn_train(training_iterations_count, batch_size)
# Final testing accuracy

print(np.mean(sess.run([corrects], feed_dict= prep_batch(final_test_data))[0]))

