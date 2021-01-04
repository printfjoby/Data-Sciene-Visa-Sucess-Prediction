# Visa Success Predection

Data set of a study abroad education consultency is used to train a model in order to predict the visa success rate. 

### Importing libraries

    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot
    from sklearn.externals import joblib`

### Read processed data file

    df=pd.read_csv('data_final.csv')
    df

  ![Input Screen](/images/1.jpg)

#### Finding Mean, Median, Standard Deviation
    df.describe().transpose()
  
  ![Input Screen](/images/2.jpg)

#### Checking for null values

    df.apply(lambda x: sum(x.isnull()),axis=0)

  ![Input Screen](/images/3.jpg)

### Finding Correlation
    cor=df.corr()
    cor
    
  ![Input Screen](/images/4.jpg)

### Ploting Heat Map

    pyplot.figure(figsize=(30,30))
    sns.heatmap(cor,annot=True,cmap="RdYlGn")
    
  ![Input Screen](/images/5.jpg)


### Checking Feature Importance

    X = df.iloc[:,0:16]  #independent columns
    y = df.iloc[:,-1]    #target column
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    pyplot.figure(figsize=(10,10))
    feat_importances.nlargest(16).plot(kind='barh')
    pyplot.show()

  ![Input Screen](/images/6.jpg)

### Feature Score

    X = df.iloc[:,0:16]  #independent columns
    y = df.iloc[:,-1]    #target column  best features
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    bestfeatures = SelectKBest(score_func=chi2, k=16)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Factors','Score']  #naming the dataframe columns
    print(featureScores.nlargest(16,'Score'))  #print 54 best features

  ![Input Screen](/images/7.jpg)


### Splitting data into train and test

    x = df.iloc[:,0:16] 
    y = df.iloc[:,-1] 
    from sklearn.model_selection import train_test_split
    #spliting test and train set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    print(x_train)
    print(y_train)

  ![Input Screen](/images/8.jpg)

## Applying Model
### Logistic Regression
    from sklearn.linear_model import LogisticRegression
    #Make an instance of the Model
    logisticRegr = LogisticRegression()

    logisticRegr.fit(x_train, y_train)
    #Predict labels for new data
    y_predict = logisticRegr.predict(x_test)

    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_predict, normalize = True)
    print("Accuracy : ",score)

  ![Input Screen](/images/9.jpg)

### Creating Confusion Matrix

    from sklearn import metrics
    cm = metrics.confusion_matrix(y_test, y_predict)
    pyplot.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    pyplot.ylabel('Actual label');
    pyplot.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    pyplot.title(all_sample_title, size = 15)

  ![Input Screen](/images/10.jpg)

### ROC Curve

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_predict)
    print('Accuracy: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    pyplot.figure(figsize=(5,5))
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    #show the plot
    pyplot.show()

  ![Input Screen](/images/11.jpg)
  
## Naive-Bayes
  
    from sklearn.naive_bayes import GaussianNB
    #create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    gnb.fit(x_train, y_train)
    y_predict_nb = gnb.predict(x_test)
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict_nb))

  ![Input Screen](/images/12.jpg)