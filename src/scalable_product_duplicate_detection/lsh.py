"""
File to do the locality sensitive hashing. Such that we only 
compare similar products instead of all. This entails:
1) Hash embedding vectors in n-length signature vector
 using random hyperplaning.
2) LSH: Split each signature vector into b bands of r bits.
Products whose bits are the same in atleast one band share
a bucket and become a candidate pair. More similar products
have a higher chance to be in the same bucket. 
"""
