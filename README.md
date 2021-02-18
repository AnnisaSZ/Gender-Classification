This project aims to provide a gender classification for twitter users. The algorithms used to build this program are BM25 and K-Nearest Neighbor (KNN). The dataset consists of 1000 tweets which are classified into 2 categories, namely women (500 data) and men (500 data).

At the testing stage, 4 types of testing are carried out:

K-Fold Cross Validation Testing: To evaluate the performance of the method used by dividing the data into two parts, namely training data and validation data. In this test the data will be multiplied by the K value and iterated by the K value. The K values ​​used in this test are 1, 3, 5, 7, 10, 20, 30, 40, and 50.
Accuracy Testing: testing the effectiveness of the system.
Precision Testing: testing the class of the data labels provided by the classifier.
F-Measure test: can be assumed as the average value of precision and recall.
Recall testing: focuses on testing the effectiveness of the classifier.

Test results:
In the tests that have been carried out, the optimal result is to use the 3-fold cross validation value because it gets the highest F-Measure value, but the highest accuracy value is obtained when using 40-fold cross validation.
