##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np


def get_retinal_seg_metrics(seg, ground, metrics=['acc', 'sens', 'spec']): 
    metrics_score = {}
    seg = np.squeeze(seg.to('cpu').detach().numpy().copy()).astype(np.int16)
    ground = np.squeeze(ground.to('cpu').detach().numpy().copy().astype(np.int16))
    # print(seg.shape, ground.shape)
    # print(np.sum(seg>1), np.sum(ground>1))

    if 'acc' in metrics:
        metrics_score['acc'] = accuracy(seg, ground)
    if 'sens' in metrics:
        metrics_score['sens'] = sensitivity(seg, ground)
    if 'spec' in metrics:
        metrics_score['spec'] = specificity(seg, ground)

    return metrics_score

def accuracy(seg, ground): 
    #computs false negative rate
    tp=np.sum(np.multiply(ground, seg), axis=(1, 2))
    _ground = (ground==0).astype(np.int16)
    _seg = (seg==0).astype(np.int16)
    tn=np.sum(_ground * _seg, axis=(1, 2))
    tot = ground.size / np.size(ground, 0)
    if tot==0:
        return 1
    else:
        return  (tp + tn) / tot

def sensitivity(seg, ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg), axis=(1, 2))
    denom=np.sum(ground, axis=(1, 2))

    sens = num / denom
    np.place(sens, sens==np.inf, 1)
    return sens

def specificity(seg, ground): 
    # print('spec')
    #computes false positive rate
    _seg = (seg==0).astype(np.int16)
    _ground = (ground==0).astype(np.int16)
    num=np.sum(_ground * _seg, axis=(1, 2))
    denom=np.sum(_ground, axis=(1, 2))
    # print(num, denom)
    spec = num / denom
    np.place(spec, spec==np.inf, 1)
    # print(spec)
    return spec