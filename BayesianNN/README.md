# Bayesian Neural Net Regression Using Renyi Divergences

I include the Bayesian NN example with some small regression datasets.

To run the code with finite alpha values, first create folder results/ to store
test outputs. Then run 
```
python split_data_train_test.py
```
in the data folder (e.g. in data/boston/) to create training/test data. Then run
```
python test_alpha.py (dataset_name) (alpha_value)
```

You can add your own dataset to data/ following the format of the included
example data. Make sure the dataset_name is the same as the folder name of your data.
