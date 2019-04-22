from sklearn import svm
import input_titanic_data
import numpy as np
import matplotlib.pyplot as plt

  # The values are encoded into the data matrix to analyze as follows:
# idx: 
#          class: 3 columns. (1st, 2nd, or 3rd class)
#          sex:   1 column.  (M / F)
#          age:   81 cols.  First col indicates age known/unknown, the following
#                 80 cols are all zero except for a 1 at the appropriate age
#          sibsp: 9 cols. 0 to 8 (flags number of siblings and parents on board)
#          parch: 10 cols. 0 to 9 (flags number of parents/children onboard)
# 104      fare:  1 col. a real number (price)
# 105:108  embarked: 3 cols. a 1 to indicate C/S/Q as embarkation point
# y is survived

x_all, y = input_titanic_data.get_titanic_all('titanic_tsmod.csv')
# linear kernel is simplest, default RBF kernel provides slightly better accuracy for the 3 parameter svm.
clf = svm.SVC(gamma='scale', kernel='linear')

# try each single parameter to predict survival...
# passenger class
passenger_class = x_all.T[:3]
clf.fit(passenger_class.T, y)
pred_val = clf.predict(passenger_class.T)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)

# passenger sex
sex = x_all.T[3].reshape(-1,1)
clf.fit(sex, y)
pred_val = clf.predict(sex)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)

# passenger age
age = x_all.T[4:85]
clf.fit(age.T, y)
pred_val = clf.predict(age.T)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)

# number of siblings aboard, plus 1 if spouse was on board
sibsp = x_all.T[85:94]
clf.fit(sibsp.T, y)
pred_val = clf.predict(sibsp.T)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)

# number of parents/children aboard
parch = x_all.T[94:104]
clf.fit(parch.T, y)
pred_val = clf.predict(parch.T)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)

# ticket fare
# fare = x_all.T[104].reshape(-1,1)
# clf.fit(fare, y)
# pred_val = clf.predict(fare)
# correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
# print(correct_percent)

# find best 3 parameters
# sex, class, age: 79%
X = np.vstack((x_all.T[:3], x_all.T[3], x_all.T[4:85])).T
clf.fit(X, y)
pred_val = clf.predict(X)
correct_percent = (np.sum(pred_val == y)/pred_val.shape[0]) * 100
print(correct_percent)