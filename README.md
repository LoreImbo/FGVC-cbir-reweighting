# AUDIO SIGNAL CLASSIFICATION

# IMAGE CLASSIFICATION

It was decided to develop different architectures to evaluate them in a comparison on the same data and get better results.
A hand-made model was initially tried to be developed, but given the poor results, it was decided to use the transfer-learning technique, which essentially allows the import of pre-trained models with their corresponding weights.

The first model uses the DenseNet201 network, of which the last 200 layers were retrained; in fact, the trainable parameters are about 9M. The model takes as input images with dimensions 224x224x3 channels, and in addition to the structure itself of the network, two Pooling layers were added. After that two layers of Dense with fz of relu activation, with relative Batch Normalization and a Dropout rate of 50%. Finally, the fully-connected layer with size 100 and softmax activation fz so as to obtain the probabilities associated with each class.

The second architecture follows the previous one with the difference that in this case it was decided to re-train all the parameters of the DenseNet, more than 20M.

Finally, the third architecture uses the same final layers seen in the first but in this case it was decided to try a different pre-trained network, namely ResNet50 of which all the parameters were re-trained.

The results of the first architecture are not satisfactory since an accuracy of almost 100% on the training set is matched by one on validation around 65%. In addition, it can be seen that from the tenth epoch onward the performance on the related datasets recedes significantly producing an overfitting effect.

With the second model, on the other hand, very satisfactory results are obtained since it achieves 80% accuracy on validation and in general we see how the growth on validation follows that on training. One could also have decided to reduce the number of training epochs but the intent was to have the model learn as many examples of the training as possible. The good performance is also confirmed by the significant color relevance of the confusion matrix.

Finally, the third model achieves probably the worst result regarding generalization over the validation set since it achieves only 60% accuracy and in general seems to suffer from overfitting.

It was then decided to choose the second model as the best and evaluate it on the test set. 
The model is confirmed to be excellent since it achieves an accuracy of 80% and a loss less than 0.9. It is then also evaluated using a single image taken from the web related to the 757-200 class of airplanes and the model confirms the excellent performance since it ranks it correctly with a probability of 83%.



# Implementation of the paper "Feature re-weighting in content-based image retrieval"


The aim of the project is to provide a human-in-the-loop re-weighting process for a CBIR process. The reference for the retrieval architecture is the following:
*Das, G., Ray, S., & Wilson, C. (2006). Feature re-weighting in content-based image retrieval. In Image and Video Retrieval: 5th International Conference, CIVR 2006, Tempe, AZ, USA, July 13-15, 2006.*

Main concepts of the paper:
- use of the previous neural network as feature extractor;
- feature normalization using 3 time standard deviation and forcing each range in the interval [0,1];
- use of weighted Minkowski distance as similarity measure;
- update of the query results according to user preferences.

Also, a simple HTML interface is available for testing using the FGVC-Aircraft-100 dataset. A DenseNet201 neural network trained on the training set is used as feature extractor. The number of similar images detected at each step is set to 20, while our interface shows only the best six.

<img src="https://user-images.githubusercontent.com/63108350/226201266-35918085-7344-42bb-b958-5d6ee4ad936c.mp4">

Three different methods are used:

### 1. Rebalancing type 1

$\large weight-type 1: w_i^{k+1}=\frac{\epsilon+\sigma_{N_r, i}^k}{\epsilon+\sigma_{r e l, i}^k}, \epsilon=0.0001$

The new weight for the i-th feature is equal to the division between the standard deviation over the 20 retrieved images and the standard deviation over the relevant images at the previous round.

### 2. Rebalancing type 2

$\large weight-type 2: w_i^{k+1}=\frac{\delta_i^k}{\epsilon+\sigma_{r e l, i}^k}$

$\large \delta_i^k=1-\frac{\sum_{l=1}^k\left|\psi_i^{l, U}\right|}{\sum_{l=1}^k\left|F_i^{l, U}\right|}$


The new weight for the i-th feature is equal to the division between the sigma quantity defined in the second formula, which depends on the dominant range, and the standard deviation over the relevant images at the previous round.

### 3. Rebalancing type 3

$\large weight-type 3: w_i^{k+1}=\delta_i^k * \frac{\epsilon+\sigma_{N_r, i}^k}{\epsilon+\sigma_{r e l, i}^k}$

The new weight for the i-th feature is equal to the delta value defined in the previous slide by the weights of type 1.

## Results

The table below contains the performance of the methods described according to the top-20 precision metric. To automatate the process, the update of the weights is done considering the labels of each image to simulate the intervention of a human.

| Iteration    | Type 1    | Type 2    | Type 3    |
| ------------ | --------- | --------- | --------- |
| Round 0      | 77.56     | ***77.56*** | ***77.56*** |
| Round 1      | 83.94     | 61.70     | 60.33     |
| Round 2      | 84.56     | 58.84     | 57.35     |
| Round 3      | 85.10     | 59.91     | 57.94     |
| Round 4      | 85.41     | 60.09     | 57.85     |
| Round 5      | ***85.54*** | 60.53     | 57.77     |

<img src="https://user-images.githubusercontent.com/63108350/226282555-16719c50-1ce9-47f3-b601-0bb5e51fd65a.jpg" width=40%>

Only the first method leads to an increase in the metric used. For this reason, it is the one used for the interface demo.
