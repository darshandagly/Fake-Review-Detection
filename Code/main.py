import sqlite3
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, pairwise_distances
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

pd.options.mode.chained_assignment = None


def load_data():
    print("Loading Data from Database")
    conn = sqlite3.connect("../Data/yelpResData.db")
    conn.text_factory = lambda x: str(x, 'gb2312', 'ignore')
    cursor = conn.cursor()

    # Table Names
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())

    # Review Table Info
    # cursor.execute("PRAGMA table_info(review)")
    # print(cursor.fetchall())

    # Count Records in Review Table
    # cursor.execute("SELECT count(1) FROM review WHERE flagged in ('Y','N')")
    # print(cursor.fetchall())

    # Create Review DataFrame
    cursor.execute(
        "SELECT reviewID, reviewerID, restaurantID, date, rating, usefulCount as reviewUsefulCount, reviewContent, flagged FROM review WHERE flagged in ('Y','N')")
    review_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Create Reviewer DataFrame
    cursor.execute("SELECT * FROM reviewer")
    reviewer_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Create Restaurant DataFrame
    cursor.execute("SELECT restaurantID, rating as restaurantRating FROM restaurant")
    restaurant_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Merge all DataFrames
    review_reviewer_df = review_df.merge(reviewer_df, on='reviewerID', how='inner')
    df = review_reviewer_df.merge(restaurant_df, on='restaurantID', how='inner')

    # Graph of Data Distribution
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.countplot(x='flagged', data=df)
    # plt.title("Count of Reviews")
    # plt.show()
    print("Data Load Complete")
    return df


def data_cleaning(df):
    print("Cleaning Data")
    # Removing \n from date field
    for i in range(len(df['date'])):
        if df['date'][i][0] == '\n':
            df['date'][i] = df['date'][i][1:]

    # Making yelpJoinDate Format Uniform
    df['yelpJoinDate'] = df['yelpJoinDate'].apply(
        lambda x: datetime.strftime(datetime.strptime(x, '%B %Y'), '01/%m/%Y'))

    # Removing emtpy cells
    if len(np.where(pd.isnull(df))) > 2:
        # TODO
        pass

    # Pre-processing Text Reviews
    # Remove Stop Words
    stop = stopwords.words('english')
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop))

    # Remove Punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: ' '.join(word for word in tokenizer.tokenize(x)))

    # Lowercase Words
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: x.lower())
    print("Data Cleaning Complete")
    return df


def feature_engineering(df):
    print("Feature Engineering: Creating New Features")
    # Maximum Number of Reviews per day per reviewer
    mnr_df1 = df[['reviewerID', 'date']].copy()
    mnr_df2 = mnr_df1.groupby(by=['date', 'reviewerID']).size().reset_index(name='mnr')
    mnr_df2['mnr'] = mnr_df2['mnr'] / mnr_df2['mnr'].max()
    df = df.merge(mnr_df2, on=['reviewerID', 'date'], how='inner')

    # Review Length
    df['rl'] = df['reviewContent'].apply(
        lambda x: len(x.split()))

    # Review Deviation
    df['rd'] = abs(df['rating'] - df['restaurantRating']) / 4

    # Maximum cosine similarity
    review_data = df

    res = OrderedDict()

    # Iterate over data and create groups of reviewers
    for row in review_data.iterrows():
        if row[1].reviewerID in res:
            res[row[1].reviewerID].append(row[1].reviewContent)
        else:
            res[row[1].reviewerID] = [row[1].reviewContent]

    individual_reviewer = [{'reviewerID': k, 'reviewContent': v} for k, v in res.items()]
    df2 = dict()
    df2['reviewerID'] = pd.Series([])
    df2['Maximum Content Similarity'] = pd.Series([])
    vector = TfidfVectorizer(min_df=0)
    count = -1
    for reviewer_data in individual_reviewer:
        count = count + 1
        # Handle Null/single review gracefully -24-Apr-2019
        try:
            tfidf = vector.fit_transform(reviewer_data['reviewContent'])
        except:
            pass
        cosine = 1 - pairwise_distances(tfidf, metric='cosine')

        np.fill_diagonal(cosine, -np.inf)
        max = cosine.max()

        # To handle reviewier with just 1 review
        if max == -np.inf:
            max = 0
        df2['reviewerID'][count] = reviewer_data['reviewerID']
        df2['Maximum Content Similarity'][count] = max

    df3 = pd.DataFrame(df2, columns=['reviewerID', 'Maximum Content Similarity'])

    # left outer join on original datamatrix and cosine dataframe -24-Apr-2019
    df = pd.merge(review_data, df3, on="reviewerID", how="left")

    df.drop(index=np.where(pd.isnull(df))[0], axis=0, inplace=True)
    print("Feature Engineering Complete")
    return df


def under_sampling(df):
    print("Under-Sampling Data")
    # Count of Reviews
    # print("Authentic", len(df[(df['flagged'] == 'N')]))
    # print("Fake", len(df[(df['flagged'] == 'Y')]))

    sample_size = len(df[(df['flagged'] == 'Y')])

    authentic_reviews_df = df[df['flagged'] == 'N']
    fake_reviews_df = df[df['flagged'] == 'Y']

    authentic_reviews_us_df = authentic_reviews_df.sample(sample_size)
    under_sampled_df = pd.concat([authentic_reviews_us_df, fake_reviews_df], axis=0)

    # print("Under-Sampled Fake", len(under_sampled_df[(under_sampled_df['flagged'] == 'Y')]))
    # print("Under-Sampled Authentic", len(under_sampled_df[(under_sampled_df['flagged'] == 'N')]))

    # Graph of Data Distribution
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.countplot(x='flagged', data=under_sampled_df)
    # plt.title("Count of Reviews")
    # plt.show()
    print("Under-Sampling Complete")
    return under_sampled_df


def semi_supervised_learning(df, model, algorithm, threshold=0.8, iterations=40):
    df = df.copy()
    print("Training "+algorithm+" Model")
    labels = df['flagged']

    df.drop(['reviewID', 'reviewerID', 'restaurantID', 'date', 'name', 'location', 'yelpJoinDate', 'flagged',
             'reviewContent', 'restaurantRating'], axis=1, inplace=True)

    train_data, test_data, train_label, test_label = train_test_split(df, labels, test_size=0.25, random_state=42)

    test_data_copy = test_data.copy()
    test_label_copy = test_label.copy()

    all_labeled = False

    current_iteration = 0

    # param_grid = {
    #     'n_estimators': [10, 500],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    #     'criterion': ['gini', 'entropy']
    # }
    # grid_clf_acc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    #
    # grid_clf_acc.fit(train_data, train_label)

    pbar = tqdm(total=iterations)

    while not all_labeled and (current_iteration < iterations):
        # print("Before train data length : ", len(train_data))
        # print("Before test data length : ", len(test_data))
        current_iteration += 1
        model.fit(train_data, train_label)

        probabilities = model.predict_proba(test_data)
        pseudo_labels = model.predict(test_data)

        indices = np.argwhere(probabilities > threshold)

        # print("rows above threshold : ", len(indices))
        for item in indices:
            train_data.loc[test_data.index[item[0]]] = test_data.iloc[item[0]]
            train_label.loc[test_data.index[item[0]]] = pseudo_labels[item[0]]
        test_data.drop(test_data.index[indices[:, 0]], inplace=True)
        test_label.drop(test_label.index[indices[:, 0]], inplace=True)
        # print("After train data length : ", len(train_data))
        # print("After test data length : ", len(test_data))
        print("--" * 20)

        if len(test_data) == 0:
            print("Exiting loop")
            all_labeled = True
        pbar.update(1)
    pbar.close()
    predicted_labels = model.predict(test_data_copy)

    # print('Best Params : ', grid_clf_acc.best_params_)
    print(algorithm + ' Model Results')
    print('--' * 20)
    print('Accuracy Score : ' + str(accuracy_score(test_label_copy, predicted_labels)))
    print('Precision Score : ' + str(precision_score(test_label_copy, predicted_labels, pos_label="Y")))
    print('Recall Score : ' + str(recall_score(test_label_copy, predicted_labels, pos_label="Y")))
    print('F1 Score : ' + str(f1_score(test_label_copy, predicted_labels, pos_label="Y")))
    print('Confusion Matrix : \n' + str(confusion_matrix(test_label_copy, predicted_labels)))
    plot_confusion_matrix(test_label_copy, predicted_labels, classes=['N', 'Y'],
                          title=algorithm + ' Confusion Matrix').show()


def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return plt


def main():
    start_time = time()
    df = load_data()
    df = data_cleaning(df)
    df = feature_engineering(df)
    # df.to_csv('df.csv', sep=',', index=False)
    under_sampled_df = under_sampling(df)
    rf = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=14, max_features='auto',
                                n_estimators=500)
    nb = GaussianNB()

    semi_supervised_learning(under_sampled_df, model=rf, threshold=0.7, iterations=15, algorithm='Random Forest')
    semi_supervised_learning(under_sampled_df, model=nb, threshold=0.7, iterations=15, algorithm='Naive Bayes')
    end_time = time()
    print("Time taken : ", end_time - start_time)


if __name__ == '__main__':
    main()
