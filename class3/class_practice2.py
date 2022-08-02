import iit_naive_bayes as iitnb
import os

# print(os.getcwd())

nb = iitnb.IITNaiveBayses("./class3/kokoro.txt", "./class3/daigakuin.txt","eng")

print(nb.predict_proba("私がその掛茶屋で先生を見た時は、先生がちょうど着物を脱いでこれから海へ入ろうとするところであった。"))

print(nb.predict_proba("修士課程は、広い視野に立つて精深な学識を授け、専攻分野における研究能力又はこれに加えて高度の専門性が求められる職業を担うための卓越した能力を培うことを目的とする。 "))

