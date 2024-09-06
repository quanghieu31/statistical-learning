These are very applied, please quickly read these:

- https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00305-w (261 citations)
- https://link.springer.com/article/10.1007/s10994-018-5724-2 (151 citations)
- https://link.springer.com/chapter/10.1007/978-3-319-30671-1_4 (220 cites)
- https://link.springer.com/article/10.1007/s00038-016-0902-0 (interactions between categorical variables, 24 citations)
- file:///C:/Users/ASUS/Downloads/paper.pdf (master's thesis but seems nice)
- https://link.springer.com/article/10.1186/s40537-020-00305-w (surveying tools to deal with cate variables in nn, 261 citations)
- https://link.springer.com/chapter/10.1007/978-981-19-6631-6_26 (10 citations, but new, modern)
- https://arxiv.org/abs/1604.06737 (fun)

Imple:
- http://vigir.ee.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_WCCI_2020/IJCNN/Papers/N-20889.pdf (LSA, GloVe)


## Optimize the Combination of Categorical Variable Encoding and Deep Learning Technique for the Problem of Prediction of Vietnamese Student Academic Performance 

A simple but interesting paper: (https://www.proquest.com/openview/0fc0f4c448e36bb4b4fcca01f05292ab/1?pq-origsite=gscholar&cbl=5444811)

- predetermined transformation methods
- algorithmic methods
- automatic transformation methods

Three common methods used are: 
1) Label Encoding; 
2) One-hot Encoding and its modification; 
3) “Learned” Embedding encoding

Combining methods of processing variables of classification data and deep learning techniques, two deep learning models are proposed for the SAPP problem. In which, 
a model uses Deep Dense network architecture (Fig. 1a), and one uses a Long short-term memory recurrent network architecture

Recommended embedding vector dim calculation:
- Emb_size = min(50, (n_cat/2)+1) (1)
- Emb_size: is size of output dimension of an embedding layer.
- n_cat: is the force of the categorical variable

## Experiment with

LSA, SVD, PCA, t-SNE for learned embedding encoding for categorical variable
(before trying NN)