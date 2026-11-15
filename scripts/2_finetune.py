"""
Fine tune embedding models per each boostrap sample.
"""

#Load bootstrap samples
#Serialize the training data, such that each index of the loaded data is serialized (the full text) That is our full training data. Then:


#For each bootstrapped sample:
  #Use training data as input to finetune_embedding(finetune_data, validation data, embedding model)
  #This should write the embedded models to the results folder, to be used in further steps.
