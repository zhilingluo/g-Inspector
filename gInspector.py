# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import time
import loader as loader

try:
    xrange
except NameError:
    xrange = range

arg = loader.parse_args()
data_name = arg.data
test_ratio = arg.test_ratio
agent = arg.agent
dataset = loader.read_data_sets(data_name, test_ratio)
neighbor_num = dataset.min_node_num
max_node = dataset.max_node_num
observation_size = dataset.node_dim
n_classes = dataset.n_classes


n_hiddens = arg.hidden_layers
instruction_length = arg.instruction_length
batch_size = arg.batch_size
max_iters = arg.iter


save_dir = "chckPts/"

initLr = 5e-4
lrDecayRate = .99
lrDecayFreq = 200
momentumValue = .9

# network units
hg_size = 128  #
hl_size = 128  #
g_size = 256  #
cell_size = 256  #
cell_out_size = 256  #
SMALL_NUM = 1e-10
max_distance = 3
siv_sd = 0.11  # std when setting the location

# resource prellocation
mean_sivs = []  # expectation of shift instruction vectors
sampled_sivs = []  # sampled siv ~N(mean_sivs[.], siv_sd)
baselines = []  # baseline, the value prediction
observed_graphs = []  # to show in window


def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
    return tf.Variable(initial, name=myname, trainable=train)


def shiftOperation(graphs, neighbors, normInstruction, curId):
    d = tf.convert_to_tensor(np.array([[max_distance for _ in range(instruction_length)] for _ in range(batch_size)]), dtype=tf.float32)
    shiftInstrutions = tf.cast(tf.round(tf.multiply(((normInstruction + 1) / 2.0), d)), tf.int32)

    curId = tf.cast(curId, tf.int32)
    graphs = tf.reshape(graphs, (batch_size, max_node, observation_size))

    zooms = []
    selectIds = []

    for k in xrange(batch_size):
        cur_id = curId[k, 0]
        instructions = shiftInstrutions[k]

        for i in range(instruction_length):
            neighbor_ids = neighbors[k, cur_id, :]
            cur_id = neighbor_ids[instructions[i]]

        graphZoom = graphs[k, cur_id, :]
        zooms.append(tf.stack(graphZoom))
        selectIds.append(tf.stack(cur_id))

    zooms = tf.stack(zooms)
    selectIds = tf.stack(selectIds)
    selectIds = tf.reshape(selectIds, shape=(batch_size, 1))
    observed_graphs.append(zooms)

    return zooms, selectIds


def getObservedFeature(instruction, curId):
    # get input using the previous location
    observation, nextId = shiftOperation(inputs_placeholder, neighbor_placeholder, instruction, curId)
    observation = tf.reshape(observation, (batch_size, observation_size))

    # the hidden units that process observation(obs) & the input
    act_obs_hidden = tf.nn.relu(tf.matmul(observation, Wi_o_h) + Bi_o_h)
    # the hidden units that process shift instruction vector(siv) & the input
    act_siv_hidden = tf.nn.relu(tf.matmul(instruction, Wi_i_h) + Bi_i_h)

    # the hidden units that integrates the shift instruction vector & the glimpses
    observedFeature = tf.nn.relu(tf.matmul(act_obs_hidden, Wi_ho_of1) + tf.matmul(act_siv_hidden, Wi_hi_of1) + Bi_hohi_of1)
    return observedFeature, nextId


def getNextObservtion(output, nextId):
    # the next shift instruction vector is computed by the shift network
    baseline = tf.sigmoid(tf.matmul(output, Wb_h_b) + Bb_h_b)
    baselines.append(baseline)
    # compute the next instruction, then impose noise
    mean_siv = tf.matmul(output, Wa_h_i)

    mean_siv = tf.stop_gradient(mean_siv)
    mean_sivs.append(mean_siv)

    # add noise
    sample_siv = tf.maximum(-1.0, tf.minimum(1.0, mean_siv + tf.random_normal(mean_siv.get_shape(), 0, siv_sd)))


    sample_siv = tf.stop_gradient(sample_siv)
    sampled_sivs.append(sample_siv)

    return getObservedFeature(sample_siv, nextId)


def affineTransform(x, output_dim):
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w) + b


# Core NetWork
def model():
    initial_siv = tf.random_uniform((batch_size, instruction_length), minval=-1, maxval=1)
    mean_sivs.append(initial_siv)
    initial_loc = tf.tanh(initial_siv + tf.random_normal(initial_siv.get_shape(), 0, siv_sd))
    sampled_sivs.append(initial_siv)

    initial_id = tf.random_uniform((batch_size, 1), minval=-1, maxval=1)
    cur_id = tf.floor(tf.multiply(((initial_id + 1) / 2.0), tf.reshape(nodenum_placeholder, shape=(batch_size, 1))))
    cur_id = tf.reshape(cur_id, shape=(batch_size, 1))
    cur_id = tf.cast(cur_id, tf.int32)

    # get the input using the input network
    initial_observation, next_id = getObservedFeature(initial_loc, cur_id)

    # set up the recurrent structure
    inputs = [0] * n_hiddens
    outputs = [0] * n_hiddens
    observation = initial_observation
    REUSE = tf.AUTO_REUSE

    for t in range(n_hiddens):
        if t == 0:  # initialize the hidden state to be the zero vector
            hiddenState_prev = tf.zeros((batch_size, cell_size))
        else:
            hiddenState_prev = outputs[t - 1]

        # forward prop
        with tf.variable_scope("coreNetwork", reuse=REUSE):
            # the next hidden state is a function of the previous hidden state and the current observed feature
            hiddenState = tf.nn.relu(affineTransform(hiddenState_prev, cell_size) + (tf.matmul(observation, Wc_o_h) + Bc_o_h))

        # save the current observed feature and the hidden state
        inputs[t] = observation
        outputs[t] = hiddenState
        # get the next observation
        if t != n_hiddens - 1:
            observation, next_id = getNextObservtion(hiddenState, next_id)
        else:
            baseline = tf.sigmoid(tf.matmul(hiddenState, Wb_h_b) + Bb_h_b)
            baselines.append(baseline)

    return outputs[-1]


def multi():
    memory = tf.zeros((batch_size, cell_size))
    for i in range(agent):
        output = model()
        memory += output
    return memory/agent


# Convert class labels from scalars to one-hot vectors
def dense_to_one_hot(labels_dense, num_classes=n_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (siv_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(siv_sd))
    return Z * tf.exp(a)


def calc_reward(outputs):
    # consider the action at the last time step
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))

    # get the baseline
    b = tf.stack(tf.reduce_mean(tf.split(baselines, agent), axis=0))
    b = tf.concat(axis=2, values=[b for _ in range(instruction_length)])
    b = tf.reshape(b, (batch_size, n_hiddens * instruction_length))
    no_grad_b = tf.stop_gradient(b)

    # get the action(classification)
    p_y = tf.nn.softmax(tf.matmul(outputs, Wa_h_c) + Ba_h_c)
    max_p_y = tf.arg_max(p_y, 1)
    correct_y = tf.cast(labels_placeholder, tf.int64)

    # reward for all examples in the batch
    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
    reward = tf.reduce_mean(R)  # mean reward
    R = tf.reshape(R, (batch_size, 1))
    R = tf.tile(R, [1, n_hiddens * instruction_length])

    p_siv= gaussian_pdf(mean_sivs, sampled_sivs)
    p_siv = tf.tanh(p_siv)
    p_siv = tf.reshape(p_siv, (batch_size, n_hiddens * instruction_length))

    J = tf.concat(axis=1, values=[tf.log(p_y + SMALL_NUM) * (onehot_labels_placeholder),
                                  tf.log(p_siv + SMALL_NUM) * (R - no_grad_b)])
    J = tf.reduce_sum(J, 1)
    J = J - tf.reduce_sum(tf.square(R - b), 1)
    J = tf.reduce_mean(J, 0)
    cost = -J

    # define the optimizer
    optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
    train_op = optimizer.minimize(cost, global_step)

    return cost, reward, max_p_y, correct_y, train_op, b, tf.reduce_mean(b), tf.reduce_mean(R - b), lr, tf.reduce_mean(
        tf.log(p_y)), tf.reduce_mean(tf.log(p_siv))



def evaluate(data):
    batches_in_epoch = len(data.graphs) // batch_size
    if batches_in_epoch < 1:
        batches_in_epoch = 1

    accuracy = 0

    for i in xrange(batches_in_epoch):
        nextX, nextY, neighbor, nodes_num = data.next_batch(batch_size)

        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, \
                     onehot_labels_placeholder: dense_to_one_hot(nextY), \
                     nodenum_placeholder: nodes_num, neighbor_placeholder: neighbor}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in_epoch
    print("ACCURACY: %.4f" % accuracy)
    return accuracy

with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

    labels = tf.placeholder("float32", shape=[batch_size, n_classes])
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels_raw")
    onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name="labels_onehot")
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, max_node, observation_size), name="graphs")
    nodenum_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="node_num")
    neighbor_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None, neighbor_num), name="node_neighbor")

    # declare the model parameters, here're naming rule:
    # the 1st captical letter: weights or bias (W = weights, B = bias)
    # the 2nd lowercase letter: the network (e.g.: i = inspection network, c = core network, a = action network)
    # the 3rd and 4th letter(s): input-output mapping, which is clearly written in the variable name argument

    Wi_i_h = weight_variable((instruction_length, hl_size), "inspectionNet_wts_instruction_hidden", True)
    Bi_i_h = weight_variable((1, hl_size), "inspectionNet_bias_instruction_hidden", True)

    Wi_o_h = weight_variable((observation_size, hg_size), "inspectionNet_wts_observation_hidden", True)
    Bi_o_h = weight_variable((1, hg_size), "inspectionNet_bias_observation_hidden", True)


    Wi_ho_of1 = weight_variable((hg_size, g_size), "inspectionNet_wts_hiddenObservation_observationFeature1", True)
    Wi_hi_of1 = weight_variable((hl_size, g_size), "inspectionNet_wts_hiddenInstruction_observationFeature1", True)
    Bi_hohi_of1 = weight_variable((1, g_size), "inspectionNet_bias_hObservation_hInstruction_observationFeature1", True)

    Wc_o_h = weight_variable((cell_size, g_size), "coreNet_wts_observation_hidden", True)
    Bc_o_h = weight_variable((1, g_size), "coreNet_bias_observation_hidden", True)


    Wb_h_b = weight_variable((g_size, 1), "baselineNet_wts_hiddenState_baseline", True)
    Bb_h_b = weight_variable((1, 1), "baselineNet_bias_hiddenState_baseline", True)


    Wa_h_i = weight_variable((cell_out_size, instruction_length), "actionNet_wts_hidden_instruction", True)

    Wa_h_c = weight_variable((cell_out_size, n_classes), "actionNet_wts_hidden_classification", True)
    Ba_h_c = weight_variable((1, n_classes), "actionNet_bias_hidden_classification", True)

    # query the model output
    if agent == 1:
        #
        outputs = model()
    else:
        outputs = multi()

    # convert list of tensors to one big tensor
    # tf.reduce_mean(tf.split(baselines, agent), axis=0)
    sampled_sivs = tf.concat(axis=0, values=tf.reduce_mean(tf.split(sampled_sivs, agent), axis=0))
    sampled_sivs = tf.reshape(sampled_sivs, (n_hiddens, batch_size, instruction_length))
    sampled_sivs = tf.transpose(sampled_sivs, [1, 0, 2])
    mean_sivs = tf.concat(axis=0, values=tf.reduce_mean(tf.split(mean_sivs, agent), axis=0))
    mean_sivs = tf.reshape(mean_sivs, (n_hiddens, batch_size, instruction_length))
    mean_sivs = tf.transpose(mean_sivs, [1, 0, 2])
    observed_graphs = tf.concat(axis=0, values=tf.reduce_mean(tf.split(observed_graphs, agent), axis=0))

    # compute the reward
    cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb, lr, logpy, logpsiv = calc_reward(
        outputs)

    ####################################### START RUNNING THE MODEL #######################################
    sess = tf.Session()
    saver = tf.train.Saver()
    b_fetched = np.zeros((batch_size, (n_hiddens) * 2))

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in xrange(max_iters):
        start_time = time.time()

        # get the next batch of examples
        nextX, nextY, neighbor, nodes_num = dataset.train.next_batch(batch_size)

        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, \
                     onehot_labels_placeholder: dense_to_one_hot(nextY), \
                     nodenum_placeholder: nodes_num, neighbor_placeholder: neighbor}

        fetches = [train_op, cost, reward, predicted_labels, correct_labels, observed_graphs, avg_b, rminusb, \
                   mean_sivs, sampled_sivs, lr]
        # feed them to the model
        results = sess.run(fetches, feed_dict=feed_dict)
        _, cost_fetched, reward_fetched, prediction_labels_fetched, correct_labels_fetched, observed_graphs_fetched, \
        avg_b_fetched, rminusb_fetched, mean_sivs_fetched, sampled_sivs_fetched, lr_fetched = results


        duration = time.time() - start_time

        if epoch % 1000 == 0:
            print('Step %d: cost = %.5f reward = %.5f (%.3f sec) b = %.5f R-b = %.5f, LR = %.5f'
                  % (epoch, cost_fetched, reward_fetched, duration, avg_b_fetched, rminusb_fetched, lr_fetched))

            saver.save(sess, save_dir + data_name + str(epoch) + ".ckpt")

    evaluate(dataset.test)

    # ckpt = tf.train.get_checkpoint_state(save_dir + data_name)
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     evaluate(dataset.test)
    # else:
    #     print("error error")

    sess.close()
