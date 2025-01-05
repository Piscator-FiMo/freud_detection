import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def showConfusionMatrix(y_test, y_pred, title):
    cfm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    #sns.set(font_scale=3)
    sns.set_theme(font_scale=3)
    sns.heatmap(cfm, ax=ax, annot=True, fmt=".0f", cmap=plt.cm.copper)
    ax.set_title(title+" \n Confusion Matrix")
    #ax.set_xticklabels(['', ''], fontsize=24, rotation=90)
    #ax.set_yticklabels(['', ''], fontsize=24, rotation=360)
    plt.show()

def showConfusionMatrix2x2(y_test, y_pred_log_reg,y_pred_knear,y_pred_svc,y_pred_tree):
    log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
    kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
    svc_cf = confusion_matrix(y_test, y_pred_svc)
    tree_cf = confusion_matrix(y_test, y_pred_tree)

    fig, ax = plt.subplots(2, 2, figsize=(22, 22))
    sns.set_theme(font_scale=3)
    sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, fmt=".0f", cmap=plt.cm.copper)
    ax[0, 0].set_title("Logistic Regression \n Confusion Matrix")
    #ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    #ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, fmt=".0f", cmap=plt.cm.copper)
    ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix")
    #ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    #ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(svc_cf, ax=ax[1][0], annot=True, fmt=".0f", cmap=plt.cm.copper)
    ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix")
    #ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    #ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(tree_cf, ax=ax[1][1], annot=True, fmt=".0f", cmap=plt.cm.copper)
    ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix")
    #ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    #ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    plt.show()

def show_correlationMatrix(df: pd.DataFrame, subsample_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    # Make sure we use the subsample in our correlation

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 20))
    sns.set_theme(font_scale=1)
    # Entire DataFrame
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
    ax1.set_title("Original Data Correlation Matrix", fontsize=14)

    sub_sample_corr = subsample_df.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax2)
    ax2.set_title('SubSample Correlation Matrix', fontsize=14)

    synth_corr = synthetic_df.corr()
    sns.heatmap(synth_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax3)
    ax3.set_title('Synthetic Data Correlation Matrix', fontsize=14)
    plt.show()

def visualize_data(df: pd.DataFrame, amount="Amount", time="Time"):
    #Create Barplot to Show Distribution of Classes
    sns.set_theme(font_scale=1)
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