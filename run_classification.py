import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from KaggleDatasetProvider import KaggleDatasetProvider


#
def visualize_data(df: pd.DataFrame, amount="Amount", time="Time"):
    #Create Barplot to Show Distribution of Classes
    sns.countplot(df, x = 'Class')
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    plt.show()

    #Create Histogram to Show Data Distribution
    sns.histplot(data=df, x=amount, bins=30)
    plt.title('Distribution of Transaction Amount', fontsize=14)
    plt.show()
    sns.histplot(data=df, x=time, bins=30)
    plt.title('Distribution of Transaction Time', fontsize=14)
    plt.show()

def scale_data(df: pd.DataFrame):
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    return df

def show_correlationMatrix(df: pd.DataFrame, new_df: pd.DataFrame):
    # Make sure we use the subsample in our correlation

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

    # Entire DataFrame
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
    ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

    sub_sample_corr = new_df.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax2)
    ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
    plt.show()

def removeOutliers(new_df: pd.DataFrame, col):
    # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
    vXX_fraud = new_df[col].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(vXX_fraud, 25), np.percentile(vXX_fraud, 75)
    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    vXX_iqr = q75 - q25
    print('iqr: {}'.format(vXX_iqr))

    vXX_cut_off = vXX_iqr * 1.5
    vXX_lower, vXX_upper = q25 - vXX_cut_off, q75 + vXX_cut_off
    print('Cut Off: {}'.format(vXX_cut_off))
    print(col, ' Lower: {}'.format(vXX_lower))
    print(col, ' Upper: {}'.format(vXX_upper))

    outliers = [x for x in vXX_fraud if x < vXX_lower or x > vXX_upper]
    print('Feature ',col,' Outliers for Fraud Cases: {}'.format(len(outliers)))
    print(col,' outliers:{}'.format(outliers))

    new_df = new_df.drop(new_df[(new_df[col] > vXX_upper) | (new_df[col] < vXX_lower)].index)
    print('----' * 44)
    return new_df

def traintest_split(df: pd.DataFrame):
    # Train Test Split
    X = df.drop('Class', axis=1)
    y = df['Class']

    # This is explicitly used for undersampling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test

def GridSearch(classifier, X_train, y_train, params):
    grid = GridSearchCV(classifier, params)
    grid.fit(X_train, y_train)
    # We automatically get the logistic regression with the best parameters.
    return grid.best_estimator_

def showConfusionMatrix(y_test, y_pred_log_reg,y_pred_knear,y_pred_svc,y_pred_tree, name=None):
    log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
    kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
    svc_cf = confusion_matrix(y_test, y_pred_svc)
    tree_cf = confusion_matrix(y_test, y_pred_tree)

    fig, ax = plt.subplots(2, 2, figsize=(22, 12))
    if name is not None:
        fig.suptitle(name, fontsize=14)

    sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper, fmt='d')
    ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
    ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper, fmt='d')
    ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
    ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper, fmt='d')
    ax[1][0].set_title("Support Vector Classifier \n Confusion Matrix", fontsize=14)
    ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper, fmt='d')
    ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
    ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    plt.show()


def run_classifiers(X_test, X_train, y_test, y_train, name=None):
    classifiers = {
        "LogisiticRegression": LogisticRegression(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    classifiers_params = {
        "LogisiticRegression": {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        "KNearest": {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        "Support Vector Classifier": {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
        "DecisionTreeClassifier": {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                                   "min_samples_leaf": list(range(5, 7, 1))}
    }
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        print("Classifiers: ", classifier.__class__.__name__, "Has a training score of",
              round(training_score.mean(), 2) * 100, "% accuracy score")
    log_reg = GridSearch(classifiers["LogisiticRegression"], X_train, y_train,
                         classifiers_params["LogisiticRegression"])
    knears_neighbors = GridSearch(classifiers["KNearest"], X_train, y_train, classifiers_params["KNearest"])
    svc = GridSearch(classifiers["Support Vector Classifier"], X_train, y_train,
                     classifiers_params["Support Vector Classifier"])
    tree_clf = GridSearch(classifiers["DecisionTreeClassifier"], X_train, y_train,
                          classifiers_params["DecisionTreeClassifier"])
    log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
    knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
    svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")
    tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
    # Logistic Regression fitted using SMOTE technique
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_knear = knears_neighbors.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_tree = tree_clf.predict(X_test)
    showConfusionMatrix(y_test, y_pred_log_reg, y_pred_knear, y_pred_svc, y_pred_tree, name)
    print('Logistic Regression:')
    print(classification_report(y_test, y_pred_log_reg))
    print('KNears Neighbors:')
    print(classification_report(y_test, y_pred_knear))
    print('Support Vector Classifier:')
    print(classification_report(y_test, y_pred_svc))
    print('Support Vector Classifier:')
    print(classification_report(y_test, y_pred_tree))


def prepare_undersampled_split(df):
    # Desribe Data
    df.describe()
    # visualize Data
    visualize_data(df)
    # Standardise Data
    scaled_df = scale_data(df)
    # Undersample Non-Fraud Transactions
    sample_df = scaled_df.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = sample_df.loc[df['Class'] == 1]
    non_fraud_df = sample_df.loc[df['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    # Normal Distributed DF
    print('No Frauds: ', new_df['Class'].value_counts()[0], ":",
          round(new_df['Class'].value_counts()[0] / len(new_df) * 100, 2),
          '% of the dataset')
    print('Frauds: ', new_df['Class'].value_counts()[1], ":",
          round(new_df['Class'].value_counts()[1] / len(new_df) * 100, 2),
          '% of the dataset')
    visualize_data(new_df, amount="scaled_amount", time="scaled_time")
    # Show Correlation Matrix, Difference between original data an normal_distribution_data
    show_correlationMatrix(df, new_df)
    # Remove Outliers from Variables with High Correlation
    no_outliers_df = removeOutliers(new_df, col='V10')
    no_outliers_df = removeOutliers(no_outliers_df, col='V12')
    no_outliers_df = removeOutliers(no_outliers_df, col='V14')
    visualize_data(no_outliers_df, amount="scaled_amount", time="scaled_time")
    X_train, X_test, y_train, y_test = traintest_split(no_outliers_df)
    return X_test, X_train, y_test, y_train
