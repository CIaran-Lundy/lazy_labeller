# lazy labeller

uses pretrained NN for feature extraction

performs umap dimension reduction

uses k-means clustering to find similar images

presents sets of similar images to user for labelling

trains NN on initial set of data

test on rest of data

returns low confidence classifications to user for labelling

retrains

loops until high confidence on all data
