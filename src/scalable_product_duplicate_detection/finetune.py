""""
File to fine tune embedding model based on
the modelIDs from the tvs (only the training set)

Get the indices for the bootstrapped samples for the train
and val split from bootstrap.py

Run the model with early stopping on the val data. That
is our finetuned model to be used for embedding.
"""