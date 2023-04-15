# BigCats-Image-Classification
* Here's a summary of the exploratory data analysis (EDA) performed on the dataset:

The dataset contains a total of 2440 images of 10 different species of big cats, with 80% of the images allocated to the training set and 20% allocated to the validation set. The images are distributed across 10 classes, one for each species of big cat.

The dataset appears to be balanced, with each class containing approximately the same number of images.

The average size of the images in the dataset is 380 x 531 pixels, with a standard deviation of 147 x 201 pixels. The minimum size of the images is 150 x 150 pixels, while the maximum size is 1192 x 794 pixels.

The dataset contains images of big cats from both the Panthera and Neofelis genera. Within the Panthera genus, the dataset includes images of the following species: African leopard, Amur leopard, Asiatic cheetah, Bengal tiger, Indochinese tiger, Jaguar, Lion, and Snow leopard. Within the Neofelis genus, the dataset includes images of the Clouded leopard and Sunda clouded leopard.

Overall, the dataset appears to be a good starting point for building a deep learning model to classify images of big cats. However, it may be necessary to preprocess the images to ensure they are all the same size and to augment the data to increase the size of the dataset and prevent overfitting.

* Preprocessing steps that were performed on the dataset:

Splitting the dataset into training, validation, and testing sets with a ratio of 0.8:0.1:0.1.
Resizing all images to 224x224 pixels to ensure consistency in size.
Converting all images to grayscale to reduce the number of channels from 3 to 1, which helps reduce the complexity of the model and training time.
Converting the categorical labels to numerical labels for use in training the model.
Normalizing the pixel values of the images to be between 0 and 1 to make them more suitable for training the model.
Augmenting the training set with random rotations, flips, and zooms to increase the size of the dataset and reduce overfitting.
These preprocessing steps were performed to ensure that the dataset is in a suitable format for training a deep learning model. By resizing, converting to grayscale, and normalizing the images, we ensure that the model is able to process the images in a consistent and efficient manner. By converting the labels to numerical format, we make it easier for the model to learn from them. Finally, by augmenting the training set, we increase the diversity of the data that the model is trained on, which can improve its accuracy and reduce overfitting.

* Summary of the training models:

The goal of training models is to create a model that can accurately predict the class of an image. To achieve this, I first preprocessed the dataset by resizing the images, converting them to grayscale, and normalizing the pixel values. then I split the data into training, validation, and test sets.

I experimented with different deep learning models, including  and MobileNetV2, and compared their performance. We used transfer learning, which involves using pre-trained models as a starting point and fine-tuning them on our dataset.

I trained the models using a categorical cross-entropy loss function and the Adam optimizer. I also used early stopping to prevent overfitting and reduce training time.

I trained a convolutional neural network with three convolutional layers, two dense layers, and a final output layer with 10 nodes corresponding to the 10 animal classes. I used the Adam optimizer and categorical cross-entropy loss function, and trained the model for 10 epochs with a batch size of 32. The model achieved a test accuracy of 0.625 or 63%.

Overall, I found that MobileNetV2 had the best performance, achieving an accuracy of 96.00% on the test set. This model was used to make predictions on new images.





