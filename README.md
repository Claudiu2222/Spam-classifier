### Algorithm Overview
Our implementation revolves around the Naive Bayes algorithm, chosen for its effectiveness in text classification and computational efficiency. We implemented Naive Bayes in Python, utilizing libraries like NumPy for mathematical operations. We also incorporated Laplace smoothing to mitigate zero-frequency issues.

### Dataset
We have worked with the Ling-Spam dataset, using the first nine folders for training and the tenth for testing. The data is pre-labeled as spam or non-spam, indicated by the file title prefix “spm”.

### Data Reading and Processing
We used Python's `os` library to navigate the filesystem and read emails. We implemented a custom `tokenize` function to split email texts into words, which served as the basis for feature extraction.

### Probability Model Construction
Our model calculates the likelihood of each word appearing in spam and non-spam emails, using a Python dictionary to store these frequencies. We applied Laplace smoothing to the frequencies to prevent zero probabilities for unseen words.

### Email Classification
We built a `classify_email` function to predict whether an email is spam based on the calculated probabilities and Bayes' theorem.

### Cross-Validation Strategy
To ensure the robustness of our model, we implemented a Leave-One-Out cross-validation function, retraining the model with each email omitted once to gauge the overall performance.

The Leave-One-Out cross-validation accuracy for our Naive Bayes classifier was `98.65%`, indicating consistent and reliable performance across different subsets of the data.

### Experimentation and Visualization
We conducted experiments with our custom Naive Bayes classifier and other algorithms like ID3, AdaBoost, and k-NN. The performance was evaluated and visualized using `matplotlib` to create comparative graphs of accuracy.


### Comparative Algorithm Performance
We also tested the dataset with other classifiers provided by the `scikit-learn` library. Their accuracies can be checked out in the `ml_practic.pdf` file
