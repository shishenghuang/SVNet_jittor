

import numpy as np 
import jittor as jt 


# def weighted_cross_entropy(output , target):

def weighted_cross_entropy( y_pred, y_true ):

    soft_max = jt.nn.Softmax(dim=1)
    p = soft_max(y_true * 2.0)
    #logp = K.log(p)
    q = soft_max(y_pred , axis=1)
    logq = jt.log(q)
    #print(p)

    #p_logp = tf.multiply(p , logp)
#    p_logq = tf.multiply(p , logq)
    p_logq = p * logq

    #w_p_logp = tf.multiply(class_weights , p_logp)
#    w_p_logq = tf.multiply(class_weights , p_logq)
    w_p_logq = class_weights * p_logq

    #loss_cross = w_p_logp - w_p_logq
    loss_cross = - tf.reduce_mean(w_p_logq, axis=-1)

    loss_cross = - w_p_logq.sum(1).mean()

    # weighted_cross_entropy
    # loss = - \sum_i (w[i] * p[i] * log(p[i])) + \sum_i ( w[i] * p[i] * log(q[i]))
    #tf.summary("p_logp=")
    return loss_cross