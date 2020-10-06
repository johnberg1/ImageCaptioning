# ImageCaptioning

This is a project implemented for the course EEE 443 - Neural Networks at Bilkent University.

In this project our aim is to develop a model which will produce a caption for a given naturalimage. This problem is particularly interesting since we can use a couple of deep learning
architectures that we have learned throughout this course together such as Convolutional
Neural Networks and Recurrent Neural Networks. In particular, we have used the ResNet 152
[1] structure to encode the given image for the CNN part, and LSTM cells are used to decode
the features to captions for the RNN part. Pre-trained word embeddings from GloVe [2] are
used in the embedding layer of the decoder. Cross entropy loss is used to evaluate the
performance of the network. The expected outcome is to reach a generative model that is as
close to a human as possible in captioning an unseen image.

**Sample Results**

Here are some of the captions generated for sample images:

![alt text](https://github.com/johnberg1/ImageCaptioning/blob/master/images/sample1.png)

![alt text](https://github.com/johnberg1/ImageCaptioning/blob/master/images/sample2.png)


**REFERENCES**

**[1]** He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2016,
doi:10.1109/cvpr.2016.90.

**[2]** Pennington, Jeffrey, et al. “Glove: Global Vectors for Word Representation.”
Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP) , 2014, doi:10.3115/v1/d14-1162.
