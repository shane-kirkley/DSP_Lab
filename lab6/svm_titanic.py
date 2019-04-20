from sklearn import svm
import input_titanic_data

x_all, y = input_titanic_data.get_titanic_all('titanic_tsmod.csv')

  # The values are encoded into the data matrix to analyze as follows:
# x_all idx: 
#          class: 3 columns. (1st, 2nd, or 3rd class)
#          sex:   1 column.  (M / F)
#          age:   81 cols.  First col indicates age known/unknown, the following
#                 80 cols are all zero except for a 1 at the appropriate age
#          sibsp: 9 cols. 0 to 8 (flags number of siblings and parents on board)
#          parch: 10 cols. 0 to 9 (flags number of parents/children onboard)
# 104      fare:  1 col. a real number (price)
# 105:108  embarked: 3 cols. a 1 to indicate C/S/Q as embarkation point

