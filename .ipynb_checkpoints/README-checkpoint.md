# VCL-PL:Semi-Supervised Learning from Noisy Web Data with Variational Contrastive Learning
**Abstract:** We address the problem of web supervised learning, in particular for face attribute classification. Web data suffers from image set noise, due to unrelated images that may be retrieved in response to the query. We propose a semi-supervised pseudo-labeling approach where the embedding space distribution is learnt via variational contrastive learning.  We  use 40 Gaussian sampling heads for the 40 attributes in the CelebA dataset and apply supervised contrastive learning over a limited amount of labelled data, to address the multi-label face attribute classification problem. Soft pseudo-labeling is then used to label the unlabelled data at  attribute level, followed by two-stage domain adaptation. We show that the proposed method using noisy web data brings improvements in accuracy over supervised multi-label face attribute classification in all experimental settings (over 2% points for very low-data setting). We suggest that learning the embedding distribution and the subsequent soft pseudo-labeling according to the nearest neighbors help in overcoming the noise in the unlabeled data. 