import iit_naive_bayes as iitnb
import os

nb = iitnb.IITNaiveBayses("./class3/machine_learning.txt", "./class3/two_dimensional_materials.txt","eng")

Sentences = "From existing online data, would be considerable those data be made available for benefits."
print(Sentences + ":\n", nb.predict_proba(Sentences))
Sentences = "Lasing might also be possible LED were to be integrated with optical cavity."
print(Sentences + ":\n", nb.predict_proba(Sentences))