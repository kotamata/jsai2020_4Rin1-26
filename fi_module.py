
# coding: utf-8

# ## 2020/2/23

# ## 仕様
# 関数
# + rf_(X_train, y_train, class or regression, estimator, random)
#  * データと問題設定と木の本数と乱数を入力
#  * 決定木のリストとランダムフォレストの特徴量重要度を返す
# + fc(X_test, y_test, DT_list, theta1, theta2)
#  * データと決定木のリストと閾値を入力
#  * 良い頻度 (good_freq)，悪い頻度 (bad_freq) とucbスコアを返す
# + fi(good_freq, good_freq_score, bad_freq, bad_freq_score, ucb)
#  * 良い頻度と悪い頻度とucbスコアを入力
#  * 変数重要度を返す

# # 必要なライブラリをインポート

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus as pdp
from sklearn.utils import resample
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import optuna


# # 関数 rf_

# In[2]:


def rf_(X_train, y_train, type_, estimators, random):
    
    if type_ == "c":
        rf = RandomForestClassifier(n_estimators = estimators, max_depth = 3,
                               random_state=random)
    elif type_ == "r":
        rf = RandomForestRegressor(n_estimators = estimators, max_depth = 3,
                               random_state=random)
    else:
        print("please input type classification => c or regression => r")
    
    rf.fit(X_train, y_train)
    
    #空のリストを作成し決定木を追加する
    DT_list = []
    
    for i in range(estimators):
        estimator = rf.estimators_[i]
        DT_list.append(estimator)
            
    return DT_list, rf.feature_importances_


# # 関数 fc

# In[3]:


def fc(X_test, y_test, DT_list, theta1, theta2):
    #出現回数の初期化
    good_count = np.array([0] * X_test.shape[1])
    bad_count = np.array([0] * X_test.shape[1])
    good_count_score = np.array([0.0] * X_test.shape[1])
    bad_count_score = np.array([0.0] * X_test.shape[1])
    
    n_over = 0   #良い木の本数
    n_under = 0 #悪い木の本数
    
    D_score = [] #決定木のスコアのリスト
    num = []   
        
    for i in range(len(DT_list)):
        DT = DT_list[i]
        score = DT.score(X_test, y_test)
        D_score.append(score)
        num.append(i)
    
    sort_score, sort_num = zip( *sorted( zip(D_score, num) ) )
    #sort_score = sorted(D_score)
    
    #順位閾値のとき
    if theta2 == 0:
        num_tree = int( theta1 * len(DT_list) )
        theta1= sort_score[len(sort_score) - num_tree]
        theta2 = sort_score[num_tree - 1]
    else:
        num_tree = len(DT_list)
    
    #決定木の数だけ実行
    for i in range(len(DT_list)):
        DT = DT_list[ sort_num[i] ]
        score = sort_score[i]
        """
        score_a = DT.score(X_test, y_test)
        if score == score_a:
            print( "ok" )
        else:
            print("bad")
        """
        #set() : 重複要素を削除
        #list() : リスト化
        f_list = list(set(DT.tree_.feature))  #DT.tree_.feature : 決定木の分割に使われた変数を取得
        f_list.remove(-2)    #葉ノードを意味する-2を削除 
        
        if score >= theta1 and i >= ( len(DT_list) - num_tree ):
            for over in f_list:
                good_count[over] += 1
                good_count_score[over] += score
            n_over += 1
        elif score <= theta2 and n_under < num_tree:
            for under in f_list:
                bad_count[under] += 1
                bad_count_score[under] += score
            n_under += 1
    
    #print(n_over)
    #print(n_under)
    
    #出現頻度の計算
    good_freq = np.array([0.0] * X_test.shape[1])
    bad_freq = np.array([0.0] * X_test.shape[1])
    good_freq_score = np.array([0.0] * X_test.shape[1])
    bad_freq_score = np.array([0.0] * X_test.shape[1])
    
    for i in range( len(good_count) ):
        if n_over != 0:
            good_freq[i] = good_count[i] / n_over
            good_freq_score[i] = good_count_score[i] / n_over
            if n_under != 0:
                bad_freq[i] = bad_count[i] / n_under
                bad_freq_score[i] = bad_count_score[i] / n_under
            else:
                bad_freq[i] = 0.0
                bad_freq_score[i] = 0.0
        else:
            good_freq[i] = 0.0
            good_freq_score[i] = 0.0
    
    #print("good count : "+ str(good_count_score) )
    #print("bad count : "+ str(bad_count_score) )
    #print("good freq : "+ str(good_freq) )
    #print("bad freq : "+ str(bad_freq) )
    
    #ucbスコアの計算
    t = n_over + n_under
    ucb = np.array([0.0] * X_test.shape[1])
    for j in range(len(good_count)):
        if (good_count[j] + bad_count[j]) == 0:
            ucb[j] = 0.0
        else:
            ucb[j] = math.sqrt( math.log(t) / (2 * ( good_count[j] + bad_count[j] ) ) ) 
    
    return good_freq, good_freq_score, bad_freq, bad_freq_score, ucb


# # 関数fi

# In[4]:


def fi(good_freq, good_freq_score, bad_freq, bad_freq_score, ucb):
    #変数重要度の初期化
    I_basis = np.array([0.0] * len(good_freq))
    I_freq = np.array([0.0] * len(good_freq))
    I_score = np.array([0.0] * len(good_freq))
    I_ucb = np.array([0.0] * len(good_freq))
    
    for i in range(len(good_freq)):
        if good_freq[i] == 0:  #出現回数が0のとき
            I_basis[i] = 0.0
            I_freq[i] = 0.0
            I_score[i] = 0.0
            I_ucb[i] = 0.0
        else:
            I_basis[i] = ( good_freq[i] / ( good_freq[i] + bad_freq[i] ) )
            I_freq[i] = I_basis[i] * good_freq[i]
            I_score[i] = (good_freq_score[i] / ( good_freq_score[i] + bad_freq_score[i] ) )
            #print(I_basis)
            I_ucb[i] = I_basis[i] - ucb[i]
            if I_ucb[i] < 0.0: #ucbスコアが負のとき
                I_ucb[i] = 0.0
            
    #正規化
    basis_sum = sum(I_basis)
    freq_sum = sum(I_freq)
    score_sum = sum(I_score)
    ucb_sum = sum(I_ucb)

    I_basis /= basis_sum
    I_freq /= freq_sum
    I_score /= score_sum
    I_ucb /= ucb_sum
    
    return I_basis, I_freq, I_score, I_ucb


# # 関数ex

# In[5]:


def ex(X, y, num, type_, estimators, theta1, theta2):
    I_rf = np.array([0.0] * X.shape[1])
    I_basis = np.array([0.0] * X.shape[1])
    I_freq = np.array([0.0] * X.shape[1])
    I_score = np.array([0.0] * X.shape[1])
    I_ucb = np.array([0.0] * X.shape[1])
    
    for i in range(num):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = 0)
        DT_list, I_rf_num = rf_(X_train, y_train, type_, estimators, i)
        good_freq, good_freq_score, bad_freq, bad_freq_score, ucb = fc(X_test, y_test, DT_list, theta1, theta2)
        I_basis_num, I_freq_num, I_score_num, I_ucb_num = fi(good_freq, good_freq_score, bad_freq, bad_freq_score, ucb)
        I_rf += I_rf_num
        I_basis += I_basis_num
        I_freq += I_freq_num
        I_score += I_score_num
        I_ucb += I_ucb_num
        
    I_rf /= num
    I_basis /= num
    I_freq /= num
    I_score /= num
    I_ucb /= num
    
    print("-" * 30)
    print("rf's feature importance")
    for j in range(len(I_rf)):
        print ("X" + str(j+1).ljust(2) + " : " + str( round(I_rf[j], 5) ))
        
    print("-" * 30)
    print("I_basis's feature importance")
    for j in range(len(I_basis)):
        print ("X" + str(j+1).ljust(2) + " : " + str( round(I_basis[j], 5) ))

    print("-" * 30)
    print("I_freq's feature importance")
    for j in range(len(I_freq)):
        print ("X" + str(j+1).ljust(2) + " : " + str( round(I_freq[j], 5) ))
    
    print("-" * 30)
    print("I_score's feature importance")
    for j in range(len(I_score)):
        print ("X" + str(j+1).ljust(2) + " : " + str( round(I_score[j], 5) ))
        
    print("-" * 30)
    print("I_ucb's feature importance")
    for j in range(len(I_ucb)):
        print ("X" + str(j+1).ljust(2) + " : " + str( round(I_ucb[j], 5) ))
        
    return I_rf, I_basis, I_freq, I_score, I_ucb


# # ex_fs

# In[1]:


def ex_fs(X, y, num, type_, estimators, theta1, theta2):
    I_rf = np.array([0.0] * X.shape[1])
    I_basis = np.array([0.0] * X.shape[1])
    I_freq = np.array([0.0] * X.shape[1])
    I_score = np.array([0.0] * X.shape[1])
    I_ucb = np.array([0.0] * X.shape[1])
    
    for i in range(num):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = 0)
        DT_list, I_rf_num = rf_(X_train, y_train, type_, estimators, i)
        good_freq, good_freq_score, bad_freq, bad_freq_score, ucb = fc(X_test, y_test, DT_list, theta1, theta2)
        I_basis_num, I_freq_num, I_score_num, I_ucb_num = fi(good_freq, good_freq_score, bad_freq, bad_freq_score, ucb)
        I_rf += I_rf_num
        I_basis += I_basis_num
        I_freq += I_freq_num
        I_score += I_score_num
        I_ucb += I_ucb_num
        
    I_rf /= num
    I_basis /= num
    I_freq /= num
    I_score /= num
    I_ucb /= num
    
    num1 = np.arange(1, X.shape[1] + 1)
    num2 = np.arange(1, X.shape[1] + 1)
    num3 = np.arange(1, X.shape[1] + 1)
    num4 = np.arange(1, X.shape[1] + 1)
    num5 = np.arange(1, X.shape[1] + 1)
    
    print("-" * 30)
    print("rf's feature importance")
    _, sort_num1 = zip( *sorted( zip(I_rf, num1) ) )
    for j in range(len(I_rf)):
        print ("X" + str(sort_num1[j]))
        
    print("-" * 30)
    print("I_basis's feature importance")
    _, sort_num2 = zip( *sorted( zip(I_basis, num2) ) )
    for j in range(len(I_rf)):
        print ("X" + str(sort_num2[j]))

    print("-" * 30)
    print("I_freq's feature importance")
    _, sort_num3 = zip( *sorted( zip(I_freq, num3) ) )
    for j in range(len(I_rf)):
        print ("X" + str(sort_num3[j]))
    
    print("-" * 30)
    print("I_score's feature importance")
    _, sort_num4 = zip( *sorted( zip(I_score, num4) ) )
    for j in range(len(I_rf)):
        print ("X" + str(sort_num4[j]))
        
    print("-" * 30)
    print("I_ucb's feature importance")
    _, sort_num5 = zip( *sorted( zip(I_ucb, num5) ) )
    for j in range(len(I_rf)):
        print ("X" + str(sort_num5[j]))
        
    return I_rf, I_basis, I_freq, I_score, I_ucb


# # 関数dt_score

# In[6]:


def dt_score(X, y, num):
    DT_score = np.zeros(100 * num)
    
    for i in range(num):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = i)
        rf = RandomForestClassifier(n_estimators = 100, max_depth = 3,
                               random_state=0)
        rf.fit(X_train, y_train)
        
        for j in range(100):
            estimator = rf.estimators_[j]
            score = estimator.score(X_test, y_test)
            DT_score[i*100 + j] = score
            
    hist, bin_edges = np.histogram(DT_score, bins=15)
    plt.title('DT score histgram')
    plt.xlabel('score')
    plt.ylabel('freqyency')
    plt.hist(DT_score, bins=15, histtype='barstacked', ec='black')
    under = DT_score[30*num]
    over = DT_score[70*num]
    print("under 30% score : " + str(under))
    print("over 30% score : " + str(over))


# # 関数fs

# In[3]:


def fs(X, y, type_, I):
    num = np.arange(len(I))
    score_f = np.array([0.0] * (len(I)+1) )
    sort_score, sort_num = zip( *sorted( zip(I, num) , reverse=True) )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = 0)
    
    X_train_f = pd.DataFrame(index=[], columns=[])
    X_test_f = pd.DataFrame(index=[], columns=[])
    
    if type_ == "c":
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                dt = DecisionTreeClassifier(max_depth = None, random_state = j)
                dt.fit(X_train_f, y_train)
                num = dt.score(X_test_f, y_test)
                score+=num
        
            score_f[ i+1 ] = score /50.0
            if i == 0:
                print(score_f[ i+1 ])
            
        
    elif type_ == "r": 
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                dt = DecisionTreeRegressor(max_depth = 7, min_samples_leaf=26, 
                                           max_leaf_nodes=45, random_state = j)     
                dt.fit(X_train_f, y_train)
                score += dt.score(X_test_f, y_test)
        
            score_f[ i+1 ] = score / 50.0
            if i == 0:
                print(score_f[ i+1 ])
        
    #x = np.arange(len(I)+1)
    #plt.plot(x, score_f, marker="o", color="red", linestyle = "--")
    
    return score_f


# # fs_r

# In[ ]:


def fs_r(X, y, type_, I):
    num = np.arange(len(I))
    score_f = np.array([0.0] * len(I) )
    sort_score, sort_num = zip( *sorted( zip(I, num) , reverse=True) )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = 0)
    
    X_train_f = pd.DataFrame(index=[], columns=[])
    X_test_f = pd.DataFrame(index=[], columns=[])
    
    if type_ == "c":
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                dt = DecisionTreeClassifier(max_depth = None, random_state = j)
                dt.fit(X_train_f, y_train)
                pred = dt.predict(X_test_f)
                num = dt.score(X_test_f, y_test)
                score+=num
        
            score_f[ i ] = 1 - (score / 50.0)
            if i == 0:
                print(score_f[ i ])
            
        
    elif type_ == "r": 
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                dt = DecisionTreeRegressor(max_depth = 7, min_samples_leaf=26, 
                                           max_leaf_nodes=45, random_state = j)   
                dt.fit(X_train_f, y_train)
                pred = dt.predict(X_test_f)
                score += mean_squared_error(y_test, pred)
        
            score_f[ i ] = score / 50.0
            if i == 0:
                print(score_f[ i ])
        
    #x = np.arange(len(I)+1)
    #plt.plot(x, score_f, marker="o", color="red", linestyle = "--")
    
    return score_f


# # fs_rf

# In[1]:


def fs_rf(X, y, type_, I):
    num = np.arange(len(I))
    score_f = np.array([0.0] * len(I) )
    score_c = np.array([0.0] * ( len(I)+1 )  )
    sort_score, sort_num = zip( *sorted( zip(I, num) , reverse=True) )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        random_state = 0)
    
    X_train_f = pd.DataFrame(index=[], columns=[])
    X_test_f = pd.DataFrame(index=[], columns=[])
    
    if type_ == "c":
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                rf = RandomForestClassifier(n_estimators=100, random_state = j)
                rf.fit(X_train_f, y_train)
                pred = rf.predict(X_test_f)
                num = rf.score(X_test_f, y_test)
                score+=num
        
            score_c[ i + 1 ] = (score / 50.0)
            if i == 0:
                print(score_c[ i+1 ])
                
            return score_c
        
    elif type_ == "r": 
        for i in range(len(I)):
            X_train_f[i] = X_train[ X_train.columns[ sort_num[ i ] ] ]
            X_test_f[i] = X_test[ X_test.columns[ sort_num[ i ] ] ]
            score = 0.0 
            for j in range(50):
                rf = RandomForestRegressor(n_estimators=100, random_state = j)   
                rf.fit(X_train_f, y_train)
                pred = rf.predict(X_test_f)
                score += mean_squared_error(y_test, pred)
        
            score_f[ i ] = score / 50.0
            if i == 0:
                print(score_f[ i ])
                
            return score_f
        
    #x = np.arange(len(I)+1)
    #plt.plot(x, score_f, marker="o", color="red", linestyle = "--")
    
    

