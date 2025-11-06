"""
EECS 445 Fall 2025

This script contains most of the work for the project. You will need to fill in every TODO comment.
"""

import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import helper

__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)

def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: DataFrame with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df_replaced = df.replace(-1,np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for var in static_variables:
        feature_dict[var] = df_replaced[df_replaced['Variable']==var]['Value'].item()
    # TODO  3) extract max of time-varying variables into feature dict
    for var in timeseries_variables:
        feature_dict["max_"+var] = df_replaced[df_replaced['Variable']==var]['Value'].max()
    return feature_dict
    
def generate_feature_vector_challenge(df: pd.DataFrame) -> dict[str, float]:
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    df_replaced = df.replace(-1,np.nan)
    df_replaced["Time"] = df_replaced["Time"].str.extract(r'^0*(\d+):').astype(int)
    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]
    
    feature_dict = {}
    # extract raw values of time-invariant variables into feature dict
    for var in static_variables:
        feature_dict[var] = df_replaced[df_replaced['Variable']==var]['Value'].item()
        
    # extract max of time-varying variables into feature dict 
    for var in timeseries_variables:
        df_select = df_replaced[df_replaced['Variable']==var]
        
        feature_dict[var + "_1"] = df_select.loc[df_select["Time"]<24,'Value'].mean()
        feature_dict[var + "_2"] = df_select.loc[df_select["Time"]>=24,'Value'].mean()

    return feature_dict


def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: (n, d) feature matrix, which could contain missing values

    Returns:
        X: (n, d) feature matrix, without missing values
    """
    X_imputed = X.copy()
    n,d = X.shape
    for i in range(d):
        mask = np.isnan(X[:,i])
        if mask.sum() > 0:
            X_imputed[mask,i] = np.nanmean(X[:,i])
    return X_imputed
    #raise NotImplementedError()  # TODO: implement


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (n, d) feature matrix

    Returns:
        X: (n, d) feature matrix with values that are normalized per column
    """
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_trans = scaler.transform(X)
    return X_trans
    #raise NotImplementedError()  # TODO: implement

def transform(X1:pd.DataFrame,
              X2:pd.DataFrame):
    threshold = 0.8
    missing_value_ratio = X1.isnull().sum()/len(X1)
    del_columns = (missing_value_ratio[missing_value_ratio > threshold]).index.tolist()
    X1 = X1.drop(columns=del_columns,axis=1)
    X2 = X2.drop(columns=del_columns,axis=1)
    
    category_col = ["Gender","ICUType"]
    num_col = list(set(X1.columns.tolist()) - set(category_col))
    
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_imputer.fit(X1[num_col])
    X1[num_col] = numerical_imputer.transform(X1[num_col])
    X2[num_col] = numerical_imputer.transform(X2[num_col])
    
    category_imputer = SimpleImputer(strategy='most_frequent')
    category_imputer.fit(X1[category_col])
    X1[category_col] = category_imputer.transform(X1[category_col])
    X2[category_col] = category_imputer.transform(X2[category_col])

    X1 = pd.get_dummies(X1,columns=category_col)
    X2 = pd.get_dummies(X2,columns=category_col)

    ## scaling techniques
    scaler = MinMaxScaler().fit(X1[num_col])
    X1[num_col] = scaler.transform(X1[num_col])
    X2[num_col] = scaler.transform(X2[num_col])

    return X1.values,X2.values,X1.columns.tolist()

def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: The name of the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel: The name of the Kernel used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(penalty=penalty,C=C,class_weight=class_weight)
        #raise NotImplementedError()
    elif loss == "squared_error":
        return KernelRidge(alpha=C,kernel=kernel,gamma=gamma)
        #raise NotImplementedError()
    else:
        raise ValueError(f"Unknown loss function: {loss}")


def cal_metric(y_true,y_pred=None,y_score=None,metric='accuracy',metric_fun_dict=None):
    if metric in ['accuracy']:
        return metric_fun_dict[metric](y_true,y_pred)
    if metric in ['precision', 'f1_score','sensitivity']:
        return metric_fun_dict[metric](y_true,y_pred,zero_division=0)
    if metric in ['auroc', 'average_precision']:
        return metric_fun_dict[metric](y_true,y_score)
    if metric in ["specificity"]:
        return metric_fun_dict[metric](y_true,y_pred,pos_label=-1,zero_division=0)
        
def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy",
    bootstrap: bool = False,
) -> float | tuple[float, float, float]:
    """
    Calculates the performance metric as evaluated on the true labels y_true versus the predicted scores from
    clf_trained and X. Returns single sample performance if bootstrap is False, otherwise returns the median
    and the empirical 95% confidence interval. You may want to implement an additional helper function to
    reduce code redundancy.

    Args:
        clf_trained: a fitted sklearn estimator
        X: (n, d) feature matrix
        y_true: (n, ) vector of labels in {+1, -1}
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1_score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
        bootstrap: whether to use bootstrap sampling for performance estimation
    
    Returns:
        If bootstrap is False, returns the performance for the specific metric. If bootstrap is True, returns
        the median and the empirical 95% confidence interval.
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    
    metric_fun = { "accuracy":accuracy_score,
                   "precision":precision_score,
                   "f1_score":f1_score,
                   "auroc":roc_auc_score,
                   "average_precision":average_precision_score,
                   "sensitivity":recall_score,
                   "specificity":recall_score
                }
    
    if not bootstrap:
        y_pred = clf_trained.predict(X)
        if isinstance(clf_trained,LogisticRegression):
            decision_score = clf_trained.decision_function(X)
        if isinstance(clf_trained,KernelRidge):
            decision_score = y_pred
            y_pred = np.where(y_pred>=0,1,-1)
            
        metric_score = cal_metric(y_true,y_pred,decision_score,metric,metric_fun)
        return metric_score
    else:
        num_bootstrap = 1000
        metric_score_list = []
        n = len(X)
        for i in range(num_bootstrap):
            idx_list = np.random.choice(n,size=n,replace=True)
            X_new = X[idx_list]
            y_new = y_true[idx_list]
            y_pred = clf_trained.predict(X_new)
            
            if isinstance(clf_trained,LogisticRegression):
                decision_score = clf_trained.decision_function(X_new)
            if isinstance(clf_trained,KernelRidge):
                decision_score = y_pred
                y_pred = np.where(y_pred>=0,1,-1)
                
            metric_score = cal_metric(y_new,y_pred,decision_score,metric,metric_fun)
            metric_score_list.append(metric_score)
            
        median = np.median(metric_score_list)
        lower_ic = np.percentile(metric_score_list,2.5)
        upper_ic = np.percentile(metric_score_list,97.5)
        return median,lower_ic,upper_ic
    
    #raise NotImplementedError()  # TODO: implement

    
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy"
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    #raise NotImplementedError()  # TODO: implement
    skf = StratifiedKFold(n_splits=k)
    result_list = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_tr,y_tr = X[train_index],y[train_index]
        X_val,y_val = X[test_index],y[test_index]
        clf_trained = clf.fit(X_tr,y_tr)
        result = performance(clf_trained,X_val,y_val,metric,bootstrap=False)
        result_list.append(result)
    result_list = np.array(result_list)
    return result_list.mean().item(),result_list.min().item(),result_list.max().item()
        
def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of true labels in {+1, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision", "sensitivity",
                and "specificity")
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # NOTE: use your cv_performance function to evaluate the performance of each classifier
    #raise NotImplementedError()  # TODO: implement
    best_result = 0
    best_C = None
    best_penalty = None
    
    for C in C_range:
        for penalty in penalties:
            logreg = LogisticRegression(penalty=penalty,C=C,solver="liblinear", fit_intercept=False,
                                       random_state=seed)
            cv_mean,cv_min,cv_max = cv_performance(logreg,X,y,metric,k)
            if cv_mean > best_result:
                best_result = cv_mean
                best_C = C
                best_penalty = penalty
    return best_C,best_penalty

def select_param_RBF(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    gamma_range: list[float],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of binary labels {1, -1}
        k: the number of folds 
        metric: the performance metric (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    #raise NotImplementedError()  # TODO: implement
    best_result = 0
    best_C = None
    best_gamma = None
    
    for C in C_range:
        for gamma in gamma_range:
            kr = KernelRidge(alpha=C,kernel="rbf",gamma=gamma)
            result = cv_performance(kr,X,y,metric,k=k)
            if result[0] > best_result:
                best_result = result[0]
                best_C = C
                best_gamma = gamma
    return best_C,best_gamma


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    
    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            #clf = get_classifier("logistic",penalty,C)
            clf = LogisticRegression(C=C,penalty=penalty,solver="liblinear") 
            # TODO: fit clf to X and y
            clf.fit(X,y)
            
            # TODO: extract learned coefficients from clf into w            
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_
            
            # TODO: count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()
 
def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: If you're having issues loading the data (e.g. your computer crashes, runs out of memory,
    #       debug statements aren't printing correctly, etc.) try setting n_jobs = 1 in get_project_data.
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!
    # Question 1
    avg = X_train.mean(axis=0).round(4).reshape(-1,1)
    quant = np.quantile(X_train,[0.25,0.75],axis=0)
    IQR = (quant[1] - quant[0]).round(4).reshape(-1,1)
    
    df_1 = pd.DataFrame(np.hstack((avg,IQR)),columns=["Mean Value","Interquartile Range"],index=feature_names)
    print("\nQuestion 1:")
    print(df_1.reset_index().rename(columns={"index":"Feature"}))
    print("*"*100)
    
    from tqdm import tqdm 
    # Question 2
    print("\nQuestion 2-c:")
    performance_dict = {}
    for metric in tqdm(metrics):
        best_C,best_penalty = select_param_logreg(X_train,y_train,
                                                    C_range=[1e-3,1e-2,1e-1,1,10,100,1000],
                                                    penalties=['l1','l2'],metric=metric,k=5)
        clf = LogisticRegression(penalty=best_penalty,C=best_C,solver="liblinear", fit_intercept=False,
                                       random_state=seed)
        cv_mean,cv_min,cv_max = cv_performance(clf,X_train,y_train,metric,k=5)
        performance_dict[metric] = [best_C,best_penalty,np.array([cv_mean,cv_min,cv_max]).round(4)]
    df_2_c = pd.DataFrame(performance_dict).T.reset_index()
    df_2_c.columns = ['Performance Measure', 'C', 'Penalty', 'Mean (Min, Max) CV Performance']
    print(df_2_c)

    df_2_c_1 = {}
    for metric in metrics:
        df_2_c_1[metric] = []
        for C in [1e-3,1e-2,1e-1,1,10,100,1000]:
            clf = LogisticRegression(penalty='l2',C=C,solver="liblinear", fit_intercept=False,
                                       random_state=seed)
            cv_mean,_,__ = cv_performance(clf,X_train,y_train,metric,k=5)
            df_2_c_1[metric].append(cv_mean)
    df_2_c_1 = pd.DataFrame(df_2_c_1).T.reset_index()
    df_2_c_1.columns = ["Metric","0.001","0.01","0.1","1","10","100","1000"]
    print(df_2_c_1.round(4))
    
    print("\nQuestion 2-d:")
    best_auc_C = 1
    best_auc_penalty = 'l2'    
    clf_auc = LogisticRegression(penalty=best_auc_penalty,C=best_auc_C,solver="liblinear")
    clf_auc.fit(X_train,y_train)

    clf_auc_performance = {}
    for metric in metrics:
        median,lower_ic,upper_ic = performance(clf_auc,X_test,y_test,metric,bootstrap=True)
        clf_auc_performance[metric] = [median,np.array([lower_ic,upper_ic]).round(4)]
    df_2_d = pd.DataFrame(clf_auc_performance).T.reset_index()
    df_2_d.columns = ['Performance Measure', 'Median', '(95% Confidence Interval)' ]
    print(df_2_d.round(4))
    
                                                       
    print("\nQuestion 2-e:")
    plot_weight(X_train,y_train,C_range=[1e-3,1e-2,1e-1,1,10,100,1000],penalties=['l1','l2'])

    print("\nQuestion 2-f:")
    clf_2_f = LogisticRegression(penalty='l1',C=1.0,solver="liblinear",fit_intercept=False,random_state=seed)
    clf_2_f.fit(X_train,y_train)
    df_2_f = pd.Series(clf_2_f.coef_.ravel(),index=feature_names).sort_values(ascending=False)
    print(df_2_f)
    print("*"*100)
    
    print("\nQuestion 3-2:")
    clf_imb = get_classifier("logistic",C=1.0,penalty='l2',class_weight={-1: 1, 1: 50})
    clf_imb.fit(X_train,y_train)
    imb_bootstrap_performance_dict = {}
    for metric in metrics:
        median,lower_ic,upper_ic = performance(clf_imb,X_test,y_test,metric,bootstrap=True)
        imb_bootstrap_performance_dict[metric] = [median,np.array([lower_ic,upper_ic]).round(4)]
    df_3_2 = pd.DataFrame(imb_bootstrap_performance_dict).T.reset_index()
    df_3_2.columns = ['Performance Measure', 'Median', '(95% Confidence Interval)', ] 
    print(df_3_2.round(4))
    
    plt.figure()
    for weight in [{-1:1,1:1},{-1:1,1:5}]:
        clf_3_3 = LogisticRegression(C=1.0,penalty='l2',class_weight=weight,fit_intercept=False)
        clf_3_3.fit(X_train,y_train)
        y_pred_prob = clf_3_3.predict_proba(X_test)
        auc_val = roc_auc_score(y_test,y_pred_prob[:,1])
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
        plt.plot(fpr, tpr, lw=2, label=str(weight)+f':{auc_val:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different Weights')
    plt.legend()
    plt.grid()
    plt.savefig("3-3.png",dpi=200)
    print("*"*100)


    print("\nQuestion 4-1-b:")
    C = 1.0
    logreg = LogisticRegression(penalty="l2", C=C, fit_intercept=False, random_state=seed)
    logreg.fit(X_train,y_train)
    kr = KernelRidge(alpha=1/(2*C), kernel="linear")
    kr.fit(X_train,y_train)

    logreg_bootstrap_performance_dict = {}
    kr_bootstrap_performance_dict = {}
    for metric in metrics:
        log_median,log_lower_ic,log_upper_ic = performance(logreg,X_test,y_test,metric,bootstrap=True)
        logreg_bootstrap_performance_dict[metric] = [np.array(log_median).round(4),np.array([log_lower_ic,log_upper_ic]).round(4)]
        kr_median,kr_lower_ic,kr_upper_ic = performance(kr,X_test,y_test,metric,bootstrap=True)
        kr_bootstrap_performance_dict[metric] = [np.array(kr_median).round(4),np.array([kr_lower_ic,kr_upper_ic]).round(4)]
        
    df_4_1_logreg = pd.DataFrame(logreg_bootstrap_performance_dict).T.reset_index()
    df_4_1_kr = pd.DataFrame(kr_bootstrap_performance_dict).T.reset_index()
    df_4_1_logreg.columns = ["Metric","Median","(95% Confidence Interval)"]
    df_4_1_kr.columns = ["Metric","Median","(95% Confidence Interval)"]
    print("\nLogistic Regression:")
    print(df_4_1_logreg)
    print("\nKernel Ridge:")
    print(df_4_1_kr)
    
    print("\nQuestion 4-2-b:")
    df_4_2_b = {}
    for gamma in [0.001,0.01,0.1,1,10,100]:
        kr_rbf = KernelRidge(alpha=1.0,kernel='rbf',gamma=gamma)
        cv_mean,cv_min,cv_max = cv_performance(kr_rbf,X_train,y_train,metric='auroc')
        df_4_2_b[gamma] = [np.array([cv_mean,cv_min,cv_max]).round(4)]
    df_4_2_b = pd.DataFrame(df_4_2_b).T.reset_index()

    df_4_2_b.columns = ["\u03B3",'Mean (Min,Max) CV Performance']
    print(df_4_2_b)

    print("\nQuestion 4-2-c:")
    df_4_2_c = {}
    best_C,best_gamma = select_param_RBF(X_train,y_train,C_range=[0.01,0.1,1.,10,100],
                                         gamma_range=[0.01,0.1,1,10],metric='auroc')
    print(f'Best C is:{best_C},best gamma is {best_gamma}')
    kr_best = KernelRidge(alpha=best_C,kernel='rbf',gamma=best_gamma).fit(X_train,y_train)
    for metric in metrics:
        kr_median,kr_lower_ic,kr_upper_ic = performance(kr_best,X_test,y_test,metric,bootstrap=True)
        df_4_2_c[metric] = [np.array(kr_median).round(4),np.array([kr_lower_ic,kr_upper_ic]).round(4)]
    df_4_2_c = pd.DataFrame(df_4_2_c).T.reset_index()
    df_4_2_c.columns = ['Performance Measure','Median Performance','(95% Confidence Interval)']
    print(df_4_2_c)
    print("*"*100)

    print('\nQuestion 5')
    X_train, y_train, X_challenge, feature_names = helper.get_challenge_data()
    param_grid = {"penalty":['l1','l2'],
             "C":[0.01,0.1,1,10,100,1000],
             "fit_intercept":[True,False],
             "class_weight":[None,'balanced']}
    clf = GridSearchCV(LogisticRegression(solver="liblinear"),param_grid=param_grid,cv=5,scoring=['roc_auc','f1'],refit='f1',n_jobs=-1)
    clf.fit(X_train,y_train)
    print("Best parameter combination:\n",clf.best_params_)
    print(f"Mean auc score:",clf.cv_results_['mean_test_roc_auc'].max().round(4))
    print(f"Mean f1 score:",clf.cv_results_['mean_test_f1'].max().round(4))

    logreg = LogisticRegression(solver="liblinear",**clf.best_params_).fit(X_train,y_train)
    print("Confusion matrix is:\n",confusion_matrix(y_train,logreg.predict(X_train)))
    y_label = logreg.predict(X_challenge).astype(int)
    y_score = logreg.decision_function(X_challenge)
    helper.save_challenge_predictions(y_label=y_label,y_score=y_score,uniqname="uniqname")

    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()

    
if __name__ == "__main__":
    main()
