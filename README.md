# Image-Classification-Using-Pre-Trained-ResNet50

In this project, we are performing image classification using a pretrained ResNet50 model. Here is a breakdown of the steps:

- We load the image data using ImageDataGenerator and create train, validation, and test generators. The data is organized in directories, and we specify the target size, color mode, batch size, and class mode for each generator.

- We use the ResNet50 model to extract features from the images. First, we load the pretrained ResNet50 model and inspect its architecture. Then, we define a new model called resnet_embedder that takes the output from the "avg_pool" layer of ResNet50 as its output. This layer has a shape of 2048, which we save for later use. We use this resnet_embedder model to predict features for the training, validation, and test images.

- The extracted features are 4D arrays, but we reshape them into 2D arrays using the reshape() function. The reshaped features will serve as the input (X) for our neural network classifier.

-  We read the class labels from the generators and one-hot encode them using the get_one_hot() function. The one-hot encoded labels will be the target (y) that our model will predict.

- We define a not-so-deep neural network model using the Sequential API from Keras. The model consists of several dense layers with different activation functions. The input shape is determined by the number of features we extracted from ResNet50, which is DIM. We print the summary of the model architecture.

- We compile the model using the compile() function. We choose the stochastic gradient descent (SGD) optimizer, categorical crossentropy loss, and accuracy as the metric to evaluate the model's performance.

- We train the model on the training data (train_X and train_y) and validate it on the validation data (val_X and val_y). We use the fit() function to train the model for a specified number of epochs. We also use the ModelCheckpoint callback to save the weights of the best model during training.

- Using the best model weights obtained during training, we load them into a new model (newmodel) and evaluate its performance on the test data (test_X and test_y). We calculate the test loss and accuracy using the evaluate() function and print the test accuracy.
