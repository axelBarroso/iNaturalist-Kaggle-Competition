# iNaturalist-Kaggle-Competition
Code submitted in the iNaturalist Kaggle Competition 

The code contains two different nets.

The first nets is the well-known inception-resnet model. The code allows to load pretrained weights from Imagenet, 
and perform transfer learning for use them in the new dataset.

The second model is a Harris corner detection model. It is a deep novel implementation of the famous Harris corner detections, 
where corner score map has been blurred in order to obtain areas of maximum activation. Those maps are convoluted with normalized images
with the objective of give maximum values on those areas, it works as an attention map in the code. 

Normalized images and Harris map images go through the inception-resnet model giving as a result the final features used for training. 

A simple convolution layer is used for the final stage of the classification. From features map to a prediction vector.

The top-5 error rate after one epoch is 
