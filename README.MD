# Summary
For this hackhathon we want to leverage the Giza stack to build a verifiable model for predicting electricity prices.

Being able to accurately predict the price of energy is an important problem in the real world, with significant
impact on energy prices for consumers and grid efficiency. There is a big market for these types of models,
with producers, providers and traders interested in having reliable forecasts for the price of energy. 
This hackhathon is an opportunity to build ML models that predict the price of energy for the next hour. In a business
scenario, it would be possible to connect to a real time data feed, and provide verifiable predictions for the 
energy price to interested parties.

# Data
The data was obtained from kaggle, and it has been pre-processed and cleaned. Many columns have also been removed, to 
keep it somewhat simple for the purposes of this hackhathon. The data can be found in the /data directory within the project. 
The jupyter notebook `exploratory data analysis` contains some exploration of the data.   

We split the data in training, validation and test set (80%, 10%, 10% respectively). The validation data is used for 
early stopping and to guide design choices (hyperparameters, layers, etc.). The test set is used only once to compare how the 
models train on unseen data. To avoid data leakage, as we are dealing with a time series task, validation 
and test set are the last 20% of the data. We also avoid shuffling the dataset.

# Models
We built and run several different models for the forecasting problem. The choice of models was motivated by the type of
data and the resources available (a middle ground between decent performance and simplicity). We use a LSTM model, 
a Fully connected model and a Random Forest model. The target is the same for all models, but each model is embedded in a 
different pipeline, as the pre-processing steps are different for each one of them.

# Results
The results can be found in the notebook `Results`. We see that the LSTM model has decent performances, but we could not 
transpile it, as it is not currently supported. The only model we could transpile was the fully connected one, which happens
to have the worst performance!

# Avenues for improvements
* Additional pre-processing and normalization of features. Normalization is not necessary for random forests,
but it helps improve the training of the deep learning models. 

* Hyperparameter search. All the models used have several hyperparameters. We tried some combinations of hyperparameters, 
but a more thorough search would likely improve the results.


If you are wondering why I am using the "we" pronoun as a solo hacker, here is the reason 
https://math.stackexchange.com/questions/1305775/why-do-single-author-math-papers-use-we-instead-of-i (my background is in Math)

