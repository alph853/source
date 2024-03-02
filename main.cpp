#include "kNN.hpp"
#include "timer.h"

void tc1()
{
    Dataset dataset;
    
    dataset.loadFromCSV("mnist.csv");
    dataset.printHead();
    dataset.printTail();
    int nRows, nCols;
    dataset.getShape(nRows, nCols);
    cout << "Shape: " << nRows << "x" << nCols << endl;

    kNN knn(5);
    Dataset X_train, X_test, y_train, y_test;
    Dataset feature = dataset.extract(0, 100, 1, -1);
    Dataset label = dataset.extract(0, 100, 0, 0);
    train_test_split(feature, label, 0.8, X_train, X_test, y_train, y_test);

    int trainImages, trainFeatures;
    int testImages, testFeatures;

    X_train.getShape(trainImages, trainFeatures);
    X_test.getShape(testImages, testFeatures);

    cout << "Train images: " << trainImages << " Train features: " << trainFeatures << endl;
    cout << "Test images: " << testImages << " Test features: " << testFeatures << endl;

    knn.fit(X_train, y_train);
    Dataset y_pred = knn.predict(X_test);
    double accuracy = knn.score(y_test, y_pred);
    cout << "Accuracy: " << accuracy << endl;
}

int main()
{
    timer t;
    tc1();
    return 0;
}

