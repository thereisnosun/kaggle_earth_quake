import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

train_df = pd.read_csv('../data/train.csv', nrows=7000000, dtype={'accoustic_data': np.int16, 'time_to_failure': np.float64})


train_ad_sample_df = train_df['acoustic_data'].values[::100]
train_ttf_sample_df = train_df['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()


plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)


def gen_features(X):
    features = []
    features.append(X.mean())
    features.append(X.std())
    features.append(X.median())
    features.append(X.max())
    features.append(X.min())
    features.append(X.kurtosis())
    features.append(X.skew())
    features.append(np.quantile(X,0.01))
    features.append(np.quantile(X,0.05))
    features.append(np.quantile(X,0.95))
    features.append(np.quantile(X,0.99))
    features.append(np.abs(X).max())
    features.append(np.abs(X).mean())
    features.append(np.abs(X).std())
    return pd.Series(features)


def create_submission(seg_id, time_to_failure):
    data_frame = pd.DataFrame({'seg_id': seg_id, 'time_to_failure': time_to_failure},
                              columns=['seg_id', 'time_to_failure'])
    data_frame.to_csv('../prediction.csv', index=False)


train = pd.read_csv('../data/train.csv', iterator=True, chunksize=150000, dtype={'acoustic_data': np.int16,
                                                                                'time_to_failure': np.float64})

X_train = pd.DataFrame()
y_train = pd.Series()
for df in train:
    new_features_chunk = gen_features(df['acoustic_data'])
    X_train = X_train.append(new_features_chunk, ignore_index=True)
    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))


print(X_train.describe())

#train_pool = Pool(X_train, y_train)
model = CatBoostRegressor(iterations=1000, loss_function='MAE', boosting_type='Ordered')
model.fit(X_train, y_train, silent=True)
#model.save_model('')
print(model.best_score_)
#model.predcit()






