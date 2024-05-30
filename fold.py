from sklearn.model_selection import RepeatedKFold
X=['shweta', 'fold' , 'make' , 'is', 'kill' ,' about' , 'to', 'for', 'the'] 
y=[ 'noun', 'adjec' , ' verb','ok' ,' noun' , 'adb', 'noun' , 'pronoun' ,' noun']
kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
X_train = []
y_train = []
X_test = []
y_test = []
#f = open("demofile3.txt", "w")
for train, test in kf.split(X):
    print("%s %s" % (train, test))
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in train:
        X_train.append( X[i])
        y_train.append( y[i])
    for i in test:
        X_test.append( X[i])
        y_test.append( y[i])
  #  for j in X_test:
  #       f.write(j)
        
 #        f.write("*******")
 #   f.write("\n")
 
#f.close()
     
