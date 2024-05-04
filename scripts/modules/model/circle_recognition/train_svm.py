import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from joblib import dump

def train_svm(train_file, kernel='linear', C=1.0, model_save_path='SVM_COLOR.joblib', scaler_save_path='scaler.joblib'):
    try:
        train_data = pd.read_csv(train_file)

        required_columns = ['diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean', 'label']
        if not all(column in train_data.columns for column in required_columns):
            raise ValueError("Input data must contain the required columns.")

        X_train = train_data[['diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean']]
        
        def label_mapper(label):
            if label in ['1e', '2e']:
                return '1-2 euros'
            elif label in ['1cts', '2cts', '5cts']:
                return 'cuivre'
            else:
                return 'jaune'
        
        y_train = train_data['label'].apply(label_mapper)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        svm_model = svm.SVC(kernel=kernel, C=C)
        svm_model.fit(X_train_scaled, y_train)

        dump(svm_model, model_save_path)
        dump(scaler, scaler_save_path)

        return svm_model
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

train_file = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data.csv'
model = train_svm(train_file)


train_file = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data.csv'
model = train_svm(train_file)

""" try:
    svm_model = load('SVM_COLOR_NORMALIZATION.joblib')
except FileNotFoundError:
    svm_model = train_svm(train_file)
 """

