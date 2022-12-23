# Casting Product's Defects Detection using Anomaly Detection

This machine learning project uses anomaly detection models to detect the submersible pump impeller casting defects through images.

Casting is a manufacturing process in which a liquid material is usually poured into a mould, which contains a hollow cavity of the desired shape, and then allowed to solidify.

Source: [Casting](https://en.wikipedia.org/wiki/Casting).

### File Guide
* Isolation Forest serves as the entry point of the project and contains feature extraction, data transformation, and IF model.
* Local Outlier Factor contains LOF model.
* One Class SVM contains one-class SVC model.
* Autoencoder contains autoencoder model (deep learning).

### Data Collection
The image dataset is obtained through Kaggle, which consists of two different types:
* 512*512 greyscale without augmentation
* 300*300 greyscale with augmentation

Source: [casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).

### Problem Statement
Even though casting technology has become better overtime, the casting process in industry is never perfect because external factors such as defects in the molding and raw materials can exist. As a result, defective casting products can be produced. Often times, it is laborious to inspect the casting products manually to separate the defective from the normal ones. What if we can automate this process? By using machine learning on images, the model can help us detect the casting products with defects.

### Feature Engineering
As the image set consists of greyscale images, the frequency distribution of the greyscale color from 0 (pure black) to 255 (pure white) is plotted for each image.
Hence, each sample consists of 256 features.

### Method of Anomaly Detection
In general, there are two different types of detecting anomalies:
* Outlier detection: The training data contains outliers which are defined as observations that are far from the others. Outlier detection estimators thus try to fit the regions where the training data is the most concentrated, ignoring the deviant observations.
* Novelty detection: The training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. In this context an outlier is also called a novelty.

Source: [2.7. Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html).

### ML Model: Isolation Forest (IF)
* Image set: 512*512 greyscale without augmentation.
* Hyperparameter tuning: number of trees.
* Outlier detection: 58% in accuracy.

### ML Model: Local Outlier Factor (LOF)
* Image set: 512*512 greyscale without augmentation.
* Hyperparameter tuning: number of neighbours.
* Outlier detection: 71% in accuracy.
* Novelty detection: 70% in accuracy.

### ML Model: One Class SVM
* Image set: 512*512 greyscale without augmentation.
* Hyperparameter tuning: nu (see explanation below).
* Outlier detection: 63% in accuracy.
* Novelty detection: 82% in accuracy.

'nu' is an upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. A margin error corresponds to a sample that lies on the wrong side of its margin boundary: it is either misclassified, or it is correctly classified but does not lie beyond the margin.

Source: [1.4.7.3. NuSVC](https://scikit-learn.org/stable/modules/svm.html#nu-svc).

### DL Model: Autoencoder
* Image set: 300*300 greyscale with augmentation (DL performs better with large number of images)
* Hyperparameter tuning: threshold (see explanation below).
* Novelty detection: 94% in accuracy.

The anomalies are detected by calculating whether the reconstruction loss is greater than a fixed threshold. For this, we will calculate the mean average error for normal samples from the training set, then classify future examples as anomalous (defective) if the reconstruction error is higher than one standard deviation from the training set.

### Conclusion
For this image set, LOF and one class SVM models have decent performance while IF does not perform well. We can see that the autoencoder model has the best performance. As it uses neural network, a lot of hidden information in the input features can be extracted and becomes a determining factor in the predictions.
