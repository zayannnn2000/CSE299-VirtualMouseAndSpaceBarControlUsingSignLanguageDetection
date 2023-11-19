import pickle

from sklearn.ensemble import RandomForestClassifier   #sykid learn use korbo and classfier hocche data gula kibhabe akta decision eh pouchabo sheita bole dibe
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))  #pickle file ta read korbo jate model.p create korte pari

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)  # when we train a model train and test thake. So train 50 ta chobi choose korbe test korar jonno

model = RandomForestClassifier()  #we used this bc overfitting solve kore (200 chobir bairer kono chobi dile chinbe na) so this classfier is best as it uses confusion matrix ise kore)

model.fit(x_train, y_train)  #classifier  er train er  chobi fit korbe from test 

y_predict = model.predict(x_test) #train er landmarks er valuer shathe test er value compare korbo

score = accuracy_score(y_predict, y_test)  #y test er score dekhbe (accuracy level)

print('{}% of samples were classified correctly !'.format(score * 100))   #train er shathe test milay accuracy percentage dekhabe. 100 multiply bc euta point hishabe thake

f = open('model.p', 'wb')     #write kore dump korchi means pass kore dicchi values gula
pickle.dump({'model': model}, f)
f.close()
