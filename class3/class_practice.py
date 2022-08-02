import iit_naive_bayes as iitnb
import os

# print(os.getcwd())

nb = iitnb.IITNaiveBayses("./class3/Mr_P.txt", "./class3/Mr_N.txt","eng")
# nb = iitnb.IITNaiveBayses("./class3/kokoro.txt", "./class3/daigakuin.txt","eng")
# nb.print_file1()
# nb.print_file2()

nb.get_vocabulary()

nb.get_words_frequency_file1()
nb.get_words_frequency_file2()

print(nb.clf.feature_log_prob_)
print(nb.predict_proba("good good bad boring"))
# print(nb.clf.feature_count_)
# print(nb.clf.feature_log_prob_)
# print(nb.clf.class_log_prior_)
