 # PossumNet

## The Neural Networks
The neural networks in this repository determine either (depending on the network) the total length of a possum or the chest and stomach length of a possum.

> The model in the **chest_and_belly_length_regression.py** file will predict both the chest length and stomach length of a possum based on the input that contains the skull width of the possum, its foot length, its eye length, and more. Given an input, the model will output a list—the first element in the list corresponds to the predicted chest length of the possum, while the second element in the list corresponding to the predicted belly length of the possum: [[*chest_length*, *belly_length*]]. Since the model is a regression model, it uses a standard stochastic gradient descent optimizer, a mean squared error loss function, and no output activation function (as is standard in regression models). The model contains an architecture consisting of:
> - 1 Batch Normalization layer
> - 1 Input layer (with 9 input neurons and a ReLU activation function)
> - 3 Hidden layers (each with 5, 4, or 3 neurons and a ReLU activation function)
> - 1 Output layer (with 2 output neurons and no activation function)
>     * There are two output neurons because the model is predicting two values: the chest length of the possum, and the belly length of the possum

> The second model, found in the **total_length_regression.py** file will predict the total length of a possum based on essentially the same input as the previous model; the only change is that the x-values in this model's dataset include chest and belly length, while the x-values in the other model don't. Since the model is a regression model, it—like the model in **chest_and_belly_length_regression.py**—uses a standard stochastic gradient descent optimizer, a mean squared error loss function, and no output activation function. The model contains an architecture consisting of: 
> - 1 Batch Normalization layer
> - 1 Input layer (with 9 input neurons and a ReLU activation function)
> - 4 Hidden layers (each with 5, 4, 3, or 2 neurons and a ReLU activation function)
> - 1 Output layer (with 1 output neuron and no activation function)

Note that each file also includes a graph that illustrates the positive difference between the model's predicted values and actual values for each input in the x-test dataset. Feel free to further tune the hyperparameters or build upon the either network!

## XGBoost Regressor
An XGBoost Regressor model is also included in the **xgb_total_length_regression.py** file to compare the neural networks to the regressor. The XGBoost Regressor has 50000 estimators and a learning rate of 0.001, coupled with early stopping based on validation sets. It should be noted that the high number of estimators will increase run time, so if you do not wish to invest the time or computer power in training a large scale XGBoost model, 4000-5000 estimators should also work fine (50000 estimators just achieves a marginally better test accuracy). 

Like the neural networks, the **xgb_total_length_regression.py** file also includes a graph that depicts the positive difference between the model's predicted values and the actual values for each input in the x-test dataset.

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/abrambeyer/openintro-possum. Credit for the dataset collection goes to **Daniel Ihenacho**, **Hamza Khan**, **Caius**, and others on *Kaggle*. It describes different aspects of different possums, including:
- Gender
- Age
- Foot length
- Total length
- Skull width
- Tail length

It should be noted that some values are marked as *NA* in the initial dataset in the *age* and *foot length* columns. In this program, these missing values are filled in with the mean of their respective columns; a missing value in the age column would be filled in with the mean of all other age values in the dataset. It should also be noted that all x-values (input values) were scaled with Scikit-Learn's **StandardScaler** before being fed to any of the models (both the neural networks and the XGBoost Regressor) during preprocessing.

## Potential Applications
The neural networks in this project could hypothetically determine the approximate chest, belly, and or total length of a possum. However, these networks are based on inputs that include many other measurements, meaning that if a person were to try and use the networks in this project it would be highly impractical; in order to recieve the predicted values from the networks, the person would have to measure multiple other aspects of the possum, and—if they are measuring all those values—they might as well measure the chest length, belly length, and total length of the possum themselves. The neural networks in this project are rather just practice in a multiple regression problem.

## Libraries
These neural networks were created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
