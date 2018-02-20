# Neural-Networks-Handwritten-Digits-Recognition
In this repository, I have implemented both regularized multi-class logistic regression and multi-class neural networks (the one-vs-all approach) to recognize handwritten digits.
For a large number of pixels (inputs), neural networks is certainly the better choice since logistic regression is essentially a linear function and using polynomial regression with a large number of pixels (features) can get computationally expensive.
My neural network has 3 layers- an input layer, a
hidden layer and an output layer. In this case, our inputs are pixel values of
digit images. Since the images are of size 20×20 (grayscale image), this gives us 400 input layer
units (excluding the extra bias unit which always outputs +1). The neural network has 25 units in the second layer (hidden layer) and 10 output units (corresponding to the 10 digit classes).
