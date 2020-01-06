---
title: Deep Learning applied to the Search for Extra-Terrestrial Intelligence
author: Naufal Said, Bagus Yusuf, Arthur Amalvy
header-include: |
	\usepackage{float}
	\let\origfigure\figure
	\let\endorigfigure\endfigure
	\renewenvironment{figure}[1][2] {
		\expandafter\origfigure\expandafter[H]
	} {
		\endorigfigure
	}
---

\newpage


# Introduction

The _SETI_ (Search for Extra-Terrestrial Intelligence) is an american set of projects started in 1960 aiming to find proof of alien intelligence by listening to space signals. Nowadays, those signals are recovered using the specially built _Allen Telescope Array_. 

SETI researchers receive a lot of signals everyday, and therefore rely on an automated system (called SonATA) to classify them. In 2017, the SETI launched an open competition asking participant to build a solution to replace SonATA by creating a software able to classify space signals in one of seven categories. The SETI also supplied competitors with datas, taking the form of synthetic signals they generated themselves. The winning team was _Eefsubsee_\footnote{https://github.com/setiQuest/ML4SETI}, with an impressive score of 94.99% accuracy.

After the competition, the competition's dataset was released on the internet, and we decided to use it to apply techniques learned in this class.


# Task

Our task is, given a signal's spectrogram\footnote{The spectrogram of a signal is a graph of its frequency versus time, the color of points usually providing the power of the signal as an additional dimension. Beware that in our case, spectrogram are actually presented with frequency in the x-axis and time in the y-axis.}, output its class as one of the seven possible ones :

* _bright pixel_ : A very powerful signal limited in time.
* _narrowband_ : A signal with a frequency varying over a constant drift rate. Such a signal may be a sign a purposely built transmitter, therefore, it is one of the most interesting ones.
* _narrowband drd_ : A signal with a frequency varying over a varying drift rate. 
* _noise_ : This signal is probably the least interesting one for the SETI researchers, and should be discarded.
* _square pulsed narrowband_ : A narrowband signal with a periodic amplitude modulation.
* _squiggle_ : A narrowband-like signal, with bounded random frequency variation.
* _squiggle square pulsed narrowband_ : A squiggle with periodic amplitude modulation.


![The seven classes of signals](./fig/classes.png)

![An example of narrowband signal](./seti-data/primary_small/train/narrowband/1012_narrowband.png)



# Dataset

The data is comprised of artificial signals generated by the SETI project. We split it as follows :

| train set | test set | validation set | total |
|:---------:|:--------:|:--------------:|:-----:|
|   5600    |    700   |       700      |  7000 |


Each of the seven class is equally represented in each set.


# Baselines

As baselines, we propose to use K Nearest Neighbours and SVM.


## K Neareast Neighbours

![KNN accuracy vs k](./fig/knn.png)

Here, we can see that our best KNN result is 19%, which makes it a pretty low baseline. However this was to be expected, as KNN usually don't perform very well in image related problems due to their complexity.


## SVM

![SVM accuracy](./fig/svm.png)

Similarly to KNN, SVM doesn't perform really well on this task.


# Model

To classify those signals, we decided to use ResNet50 v1. ResNet50 is a very deep neural network, consisting of around 50 convolution and pooling layers.


## Architecture

![The architecture of ResNet50](./fig/resnet.png)

ResNet is composed of 4 main stages, each acting as a bottleneck : after each one, the input height and width decrease, while its depth increases. Each stage contains 3 to 6 residual blocks, each comprised of three convolution layers of respective dimensions 1x1, 3x3 and 1x1. 

The first 1x1 layer task is to reduce the depth of the input. By doing so, it reduces hte number of overall computations, by reducing the numbers of parameters of the 3x3 layer. Finally, the last 1x1 layer restores the original dimension.


## Residual blocks

To reduce the general problems around deep neural networks, ResNet authors introduced the _residual block_ concept.

![An example of residual block](./fig/residual.png)

A residual block is composed of classic deep-learning neural networks layers (in the case of ResNet, convolution layers), but the input of these layers is fast-forwarded to their output, so that the output of a residual block $H$ with layers represented by their computed function $F$ is :

$$H(x) = F(x) + x$$

which means the residual block layers are trying to learn the function

$$F(x) = H(x) - x$$

This function is perfectly learnable by a neural network, and doesn't add any complexity.  Moreover, it is a solution to the vanishing of gradient in deep neural networks. Summing the output of $F$ and its input ensures that the gradient will get distributed to the previous layer during backpropagation as $\nabla H(x) = \nabla F(x) + \nabla x$.


## Data augmentation

To increase our performace, we use _image data augmentation_ : We generate new, plausible samples from the training datas. To do that, we use the _ImageDataGenerator_ class from Keras to randomly apply transformations on the training datas : flipping, adding noise, shifting...


# Results

+-----------------------+----------+
|      Model            | Accuracy |
+:======================+:========:+
| KNN (k=1)             |   19.14  | 
+-----------------------+----------+
| Linear SVM            |   14.57  |
+-----------------------+----------+
| Sigmoid SVM           |   12.71  |
+-----------------------+----------+
| Polynomial SVM        |   11.00  |
+-----------------------+----------+
| RBF SVM               |   12.29  |
+-----------------------+----------+
| _ResNet50 (ours)_     |  _89.00_ |
+-----------------------+----------+
| **Eefsubsee**         | **94.99**|
+-----------------------+----------+

![Model accuracy over time](./fig/modelAcc.png)

![Model loss over time](./fig/modelLoss.png)


# Conclusion

Our model performance largely outperforms our weak baselines, but falls short of Eesubsee impressive work. Still, we're quite happy about the results.