import os
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

train_emails = {}
trained_words = {}
test_emails = {}
ham_emails = 0
spam_emails = 0
total_number_of_ham_words = 0
total_number_of_spam_words = 0


def tokenize(text: str):
    return text.lower().split(" ")


def apply_laplace(trained_words):
    for word in trained_words:
        trained_words[word][0] += 1
        trained_words[word][1] += 1
    return trained_words


def sort():
    global trained_words
    trained_words = dict(sorted(trained_words.items(
    ), key=lambda item: item[1][0] + item[1][1], reverse=True))


def remove_unnecessary_words():
    new_trained_words = {}
    global trained_words
    for word in trained_words:
        if len(word) > 1:
            new_trained_words[word] = trained_words[word]

    trained_words = new_trained_words


def read_training_emails(path):
    global train_emails, trained_words, ham_emails, spam_emails
    for filename in os.listdir(path):
        if filename.endswith("txt"):
            file_path = os.path.join(path, filename)
            with open(file_path) as f:
                email = f.read()
                temp_words = tokenize(email)
                if "spm" in filename:
                    train_emails[email] = 1
                    spam_emails += 1
                    for word in temp_words:
                        if word in trained_words:
                            trained_words[word][0] += 1
                        else:
                            trained_words[word] = [1, 0]
                else:
                    train_emails[email] = 0
                    ham_emails += 1
                    for word in temp_words:
                        if word in trained_words:
                            trained_words[word][1] += 1
                        else:
                            trained_words[word] = [0, 1]


def read_test_emails(path):
    for filename in os.listdir(path):
        if filename.endswith("txt"):
            file_path = os.path.join(path, filename)
            with open(file_path) as f:
                if "spm" in filename:
                    test_emails[f.read()] = 1
                else:
                    test_emails[f.read()] = 0


def print_in_wordsc(to_write):
    # deschide fisierul "wordsc.txt" si scrie in el
    with open("wordsc1.txt", "w") as f:
        f.write(to_write)


def calculate_total_number_of_words(trained_words):
    total_number_of_ham_words = 0
    total_number_of_spam_words = 0
    for word in trained_words:
        total_number_of_spam_words += trained_words[word][0]  # spam
        total_number_of_ham_words += trained_words[word][1]  # ham
    return total_number_of_ham_words, total_number_of_spam_words


def compute_conditional_probs(trained_words, total_number_of_ham_words, total_number_of_spam_words):
    for word in trained_words:
        trained_words[word][0] /= total_number_of_spam_words
        trained_words[word][1] /= total_number_of_ham_words
    return trained_words


def classify_email(email, trained_words, p_spam, p_ham):
    words = tokenize(email)
    p_spam_given_email = np.log(p_spam)
    p_ham_given_email = np.log(p_ham)
    for word in words:
        if word in trained_words:
            p_spam_given_email += np.log(trained_words[word][0])
            p_ham_given_email += np.log(trained_words[word][1])
    if p_spam_given_email > p_ham_given_email:
        return 1
    else:
        return 0


def calculate_accuracy(test_emails, trained_words, p_spam, p_ham):
    correct = 0
    for email in test_emails:
        if classify_email(email, trained_words, p_spam, p_ham) == test_emails[email]:
            correct += 1
    return correct / len(test_emails)


# training
dataset_name = "lemm"
path = f"input/{dataset_name}/"
read_training_emails(path + "train")


# ALGORITMUL MARE NU PUNE MANA CA FRIGE !!!!!!!!!
# # sort()
trained_words = apply_laplace(trained_words)
total_number_of_ham_words, total_number_of_spam_words = calculate_total_number_of_words(
    trained_words)

p_ham = ham_emails / (ham_emails + spam_emails)
p_spam = spam_emails / (ham_emails + spam_emails)
trained_words = compute_conditional_probs(
    trained_words, total_number_of_ham_words, total_number_of_spam_words)

print_in_wordsc(str(trained_words))


# testing
read_test_emails(path + "test")

custom_naive_acuracy = calculate_accuracy(
    test_emails, trained_words, p_spam, p_ham)
print(f"Accuracy: {custom_naive_acuracy * 100:.2f}")


def cross_leave_one_out():
    correct = 0
    indx = 0
    for email_to_skip in train_emails.keys():

        trained_words = {}
        spam_emails = 0
        ham_emails = 0

        total_number_of_ham_words = 0
        total_number_of_spam_words = 0
        p_spam = 0
        p_ham = 0
        # reantrenare pe setu fara email_to_skip
        for email in train_emails.keys():
            if email != email_to_skip:
                if train_emails[email] == 1:
                    spam_emails += 1
                else:
                    ham_emails += 1
                words = tokenize(email)
                for word in words:
                    if word in trained_words:
                        if train_emails[email] == 1:
                            trained_words[word][0] += 1
                        else:
                            trained_words[word][1] += 1
                    else:
                        if train_emails[email] == 1:
                            trained_words[word] = [1, 0]
                        else:
                            trained_words[word] = [0, 1]

        # aplicare laplaace
        trained_words = apply_laplace(trained_words)

        # calculare total number of words
        total_number_of_ham_words, total_number_of_spam_words = calculate_total_number_of_words(
            trained_words)

        # calculare probabilitati
        p_ham = ham_emails / (ham_emails + spam_emails)
        p_spam = spam_emails / (ham_emails + spam_emails)
        trained_words = compute_conditional_probs(
            trained_words, total_number_of_ham_words, total_number_of_spam_words)

        if classify_email(email_to_skip, trained_words, p_spam, p_ham) == train_emails[email_to_skip]:
            correct += 1
        indx += 1
        if indx % 100 == 0:
            print("Am terminat de clasificat emailul cu indexul: ", indx)
            print(f"Acuratete {correct / len(train_emails) * 100:.2f}")
    return correct / len(train_emails)


cvloo_accuracy = cross_leave_one_out()
print(f"CVLOO Accuracy: {cvloo_accuracy * 100:.2f}")


# Now test with knn, adaboost and id3
def vectorize_emails(emails):
    vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize)
    return vectorizer.fit_transform(emails), vectorizer


x_train, vectorizer = vectorize_emails(list(train_emails.keys()))
y_train = list(train_emails.values())

x_test = vectorizer.transform(list(test_emails.keys()))
y_test = list(test_emails.values())

id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(x_train, y_train)
id3_accuracy = id3_model.score(x_test, y_test)
print(f"ID3 Accuracy: {id3_accuracy * 100:.2f}%")

# AdaBoost
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(x_train, y_train)
adaboost_accuracy = adaboost_model.score(x_test, y_test)
print(f"AdaBoost Accuracy: {adaboost_accuracy * 100:.2f}%")

# k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
knn_accuracy = knn_model.score(x_test, y_test)
print(f"k-NN Accuracy: {knn_accuracy * 100:.2f}%")

# Naive Bayes
nb_model = DecisionTreeClassifier(criterion='entropy')
nb_model.fit(x_train, y_train)
nb_accuracy = nb_model.score(x_test, y_test)
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")


accuracies = {
    'ID3': id3_accuracy * 100,
    'AdaBoost': adaboost_accuracy * 100,
    'k-NN': knn_accuracy * 100,
    'Custom Naive Bayes': custom_naive_acuracy * 100,
    'CVLOO Custom Naive Bayes': cvloo_accuracy * 100
}


bars = plt.bar(range(len(accuracies)), list(
    accuracies.values()), align='center', color='skyblue')
plt.xticks(range(len(accuracies)), list(
    accuracies.keys()), rotation='vertical')
plt.ylabel('Accuracy (%)')
plt.title('Accuracies for dataset ' + dataset_name)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2),
             va='bottom', ha='center', color='black')

plt.show()
