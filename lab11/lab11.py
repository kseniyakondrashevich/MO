import os
import sys

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline

sys.path.insert(0, 'pypuf/')

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n=64, k=2),
    transform=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor
)

train_set = tools.TrainingSet(instance=instance, N=15000)
val_set = train_set.subset(slice(10000, None))
train_set = train_set.subset(slice(None, 10000))

challenges, responses = train_set.challenges, train_set.responses

print(challenges.shape, responses.shape)

from pypuf.learner.regression.logistic_regression import LogisticRegression

lr_learner = LogisticRegression(
    t_set=train_set,
    n=64,
    k=2,
    transformation=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

model = lr_learner.learn()

val_set_predicted_responses = model.eval(val_set.challenges)
acc = accuracy_score(val_set_predicted_responses, val_set.responses)

print('accuracy: ', acc)

from pypuf.learner.other import Boosting

boosting_learner = Boosting(
    t_set=train_set,
    n=64,
    k=1,
    transformation=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

model_boosting = boosting_learner.learn()

val_set_predicted_responses = model_boosting.eval(val_set.challenges)

acc = round(accuracy_score(val_set_predicted_responses, val_set.responses), 4)

print('accuracy for gradient boosting: ', acc)

from pypuf.learner.other import SVM

svm_learner = SVM(
    t_set=train_set,
    n=64,
    k=1,
    transformation=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

model_svm = svm_learner.learn()

val_set_predicted_responses = model_svm.eval(val_set.challenges)

acc = round(accuracy_score(val_set_predicted_responses, val_set.responses), 4)

print('accuracy for SVM: ', acc)

from pypuf.learner.regression.logistic_regression import LogisticRegression

lr_learner = LogisticRegression(
    t_set=train_set,
    n=64,
    k=2,
    transformation=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

model = lr_learner.learn()

val_set_predicted_responses = model.eval(val_set.challenges)

acc = accuracy_score(val_set_predicted_responses, val_set.responses)

print('accuracy for logistic regression: ', acc)

def pipeline(N):
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n=64, k=2),
        transform=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor
    )

    train_set = tools.TrainingSet(instance=instance, N=N)

    train_size = int(len(train_set.challenges) * 0.95)

    val_set = train_set.subset(slice(train_size, None))
    train_set = train_set.subset(slice(None, train_size))
    
    lr_learner = LogisticRegression(
        t_set=train_set,
        n=64,
        k=2,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    model = lr_learner.learn()
    
    val_set_predicted_responses = model.eval(val_set.challenges)

    accuracy = accuracy_score(val_set_predicted_responses, val_set.responses)
    
    return accuracy

N2accuracy = [(N, pipeline(N)) for N in [10, 50, 100, 500, 1000, 1500, 2000, 3000, 5000]]

plt.figure(figsize=(16,10))
plt.plot(*zip(*N2accuracy))

plt.xlabel('number of examples', size=20)
plt.ylabel('accuracy', size=20)
plt.show()