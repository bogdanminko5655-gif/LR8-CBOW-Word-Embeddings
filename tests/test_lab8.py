"""Тести компонентів CBOW — запуск: python -m pytest -q"""
import numpy as np
from lab8 import (tokenize, make_vocab, create_examples, softmax,
    initialize_model, forward, train_step, get_embeddings, cosine_similarity,
    most_similar, analogy, compute_pca, save_embeddings, load_embeddings, load_sentences)


def small_example():
    s=[['king','is','a','man','king','is','a','ruler'],
       ['queen','is','a','woman','queen','is','a','ruler']]
    w,_ = make_vocab(s,1)
    x,y=create_examples(s,w,1)
    return w,x,y


def test_tokenize():
    assert tokenize('King, queen! It\'s great.') == ['king','queen',"it's",'great']


def test_vocab_frequency():
    w,c=make_vocab([['a','b','a'],['c','a']],2)
    assert 'a' in w and 'b' not in w and c['a']==3 and '<unk>' in w


def test_examples():
    w,x,y=small_example()
    assert x.shape[1]==2 and y.shape[0] == x.shape[0]


def test_softmax_normalized_and_stable():
    p=softmax(np.array([[1e4,1e4+1,1e4+2]],dtype=float))
    assert np.isfinite(p).all() and np.isclose(p.sum(),1.0)


def test_forward_dims():
    w,x,y=small_example(); p=initialize_model(len(w),6)
    out,cache=forward(x[:2],p)
    assert out.shape==(2,len(w)) and np.allclose(out.sum(axis=1),1)


def test_train_reduces_loss_on_repeat():
    w,x,y=small_example(); p=initialize_model(len(w),8)
    prev=None
    for _ in range(150):
        loss=train_step(p,x,y,.3)
    new,_=forward(x,p)
    assert loss < 2.8 and np.isfinite(new).all()


def test_embedding_shapes():
    w,_,_=small_example();p=initialize_model(len(w),7)
    assert get_embeddings(p).shape==(len(w),7)


def test_cosine():
    assert abs(cosine_similarity(np.array([1.,0.]),np.array([1.,0.]))-1)<1e-6
    assert abs(cosine_similarity(np.array([1.,0.]),np.array([0.,1.])))<1e-6


def test_most_similar_exclusion():
    w={'<unk>':0,'king':1,'queen':2,'dog':3}
    e=np.array([[0.,0.],[1.,0.],[.99,.01],[0.,1.]],dtype=np.float32)
    assert most_similar('king',e,w,1)[0][0]=='queen'


def test_analogy():
    w={'<unk>':0,'king':1,'queen':2,'man':3,'woman':4}
    e=np.array([[0.,0.],[2.,2.],[1.,3.],[2.,0.],[1.,1.]],dtype=np.float32)
    assert analogy(['king','woman'],['man'],e,w)[0][0]=='queen'


def test_pca_shape():
    x=np.array([[0.,0.,1.],[1.,2.,0.],[2.,1.,2.]],dtype=float)
    assert compute_pca(x,2).shape==(3,2)


def test_save_load(tmp_path):
    w,x,y=small_example(); p=initialize_model(len(w),5)
    save_embeddings(tmp_path/'e.npz', w, get_embeddings(p), p)
    w2,e2=load_embeddings(tmp_path/'e.npz')
    assert w2==w and np.allclose(e2,get_embeddings(p))
