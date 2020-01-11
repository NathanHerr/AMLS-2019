
from A1.A1_module import A1
from A2.A2_module import A2
from B1.B1_module import B1
from B2.B2_module import B2


# ======================================================================================================================
# Task A1

A1_Model = A1()                # Build model object.
print("Training Model for Task A1")
acc_A1_train, clf = A1_Model.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
print("Testing Model for Task A1")
acc_A1_test = A1_Model.test(clf)     # Test model based on the test set.
print(acc_A1_train)
print(acc_A1_test)

# ======================================================================================================================
# Task A2

model_A2 = A2()
print("Training Model for Task A2")
acc_A2_train, clf = model_A2.train()
print("Testing Model for Task A2")
acc_A2_test = model_A2.test(clf)
print(acc_A2_train)
print(acc_A2_test)

# ======================================================================================================================
# Task B1

model_B1 = B1()
print("Training Model for Task B1")
acc_B1_train, clf = model_B1.train()
print("Testing Model for Task B1")
acc_B1_test = model_B1.test(clf)
print(acc_B1_train)
print(acc_B1_test)

# ======================================================================================================================
# Task B2

model_B2 = B2()
print("Training Model for Task B2")
acc_B2_train, clf = model_B2.train()
print("Testing Model for Task B2")
acc_B2_test = model_B2.test(clf)
print(acc_B2_train)
print(acc_B2_test)

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test, acc_A2_train, acc_A2_test, acc_B1_train, acc_B1_test, acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'