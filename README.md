Each of the experiments can be run using variants of the following command:

`python active_learning.py --dataset stream --n_samples 600 --prop_rare 0.01`

where `dataset` refer to the word being tested, `n_samples` gives the number of samples collected by the active learning algorithm and `prop_rare` gives the proportion of rare examples (e.g. 0.01 will give a 1:100 skew ratio).

You will need to download the data at the following [anonymous Google drive](https://drive.google.com/drive/folders/1LUUN8PIXnP2DOwsRU_-C91NBFFVoPFmF?usp=sharing) to run the experiments. The data just contains the embeddings and not the raw text because the embedding alone for these experiments are 9GB. 


