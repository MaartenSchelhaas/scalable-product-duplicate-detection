"""
Detect duplicate TVs per bootstrap and score the results

clean and serialize all products (once)
for each of the 5 bootstraps:
    load the split and the model of this bootstrap
    embed all products, hash to signatures (once)
    for each of the 19 values of t:
        (b, r) = closest to t
        LSH on the tuning products   -> tuning candidates
        choose epsilon: best F1 on the tuning set (one pass)
        LSH on the test products     -> test candidates
        cluster the test set with that epsilon
        save: b, r, epsilon, candidates, predicted pairs, timings
"""