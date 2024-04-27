import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump, load

def train_svm(train_file):
    train_data = pd.read_csv(train_file)

    X_train = train_data[['diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean']]
    y_train = train_data['label'].apply(lambda x: '1-2 euros' if x in ['1e', '2e'] else ('cuivre' if x in ['1cts', '2cts', '5cts'] else 'jaune'))

    svm_model = svm.SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)

    dump(svm_model, 'svm_model.joblib')

    return svm_model

train_file = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data.csv'

try:
    svm_model = load('svm_model_color.joblib')
except FileNotFoundError:
    svm_model = train_svm(train_file)


new_data = [[140, 90.14, 141.24, 148.2]]  
prediction = svm_model.predict(new_data)
print("Prédiction :", prediction)
