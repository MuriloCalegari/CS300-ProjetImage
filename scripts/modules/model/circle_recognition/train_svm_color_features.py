import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from joblib import dump
import numpy as np

def train_svm(train_file, kernel='linear', C=1.0, batch_size=100, model_save_path='SVM_COLOR_AND_TEXTURE.joblib', scaler_save_path='scaler_color_and_texture.joblib'):
    try:
        reader = pd.read_csv(train_file, chunksize=batch_size)
        
        scaler = StandardScaler()
        svm_model = svm.SVC(kernel=kernel, C=C)

        first_batch = True
        for batch in reader:
         
            batch.dropna(inplace=True)

            required_columns = ['diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean', 'label', 'hog_features']
            if not all(column in batch.columns for column in required_columns):
                raise ValueError("Input data must contain the required columns.")
            
            X_batch = batch[['diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean', 'hog_features']].copy()
            X_batch.columns = X_batch.columns.astype(str)  
            
            y_batch = batch['label'].apply(lambda label: '1-2 euros' if label in ['1e', '2e'] else 'cuivre' if label in ['1cts', '2cts', '5cts'] else 'jaune')

            X_batch['hog_features'] = X_batch['hog_features'].apply(lambda x: list(map(float, x.split(','))))
            hog_features_list = X_batch.pop('hog_features').tolist()
            hog_features_df = pd.DataFrame(hog_features_list)
            hog_features_df.columns = ['hog_feature_' + str(i) for i in range(hog_features_df.shape[1])]

            X_full_batch = pd.concat([X_batch, hog_features_df], axis=1)

            if first_batch:
                X_scaled_batch = scaler.fit_transform(X_full_batch)
                first_batch = False
            else:
                X_scaled_batch = scaler.transform(X_full_batch)

            if np.isnan(X_scaled_batch).any():
                continue 
            
            svm_model.fit(X_scaled_batch, y_batch)
        
        dump(svm_model, model_save_path)
        dump(scaler, scaler_save_path)
        
        return svm_model
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

train_file = '/Volumes/SSD/ProjetImage/testcombined.csv'
model = train_svm(train_file)
