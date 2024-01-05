import os
from collections import Counter
import numpy as np


trained_words = {}
test_emails = {}
train_emails = {}
spam_emails = 0
ham_emails = 0

total_number_of_ham_words = 0
total_number_of_spam_words = 0
p_spam = 0
p_ham = 0


def tokenize(text: str):
    return text.lower().split(" ")


def apply_laplace():
    for word in trained_words:
        trained_words[word][0] += 1
        trained_words[word][1] += 1


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
    for filename in os.listdir(path):
        if filename.endswith("txt"):
            file_path = os.path.join(path, filename)
            with open(file_path) as f:
                temp_words = tokenize(f.read())
                if "spm" in filename:
                    global spam_emails
                    spam_emails += 1
                    train_emails[f.read()] = 1
                    for word in temp_words:
                        if word in trained_words:
                            trained_words[word][0] += 1
                        else:
                            trained_words[word] = [1, 0]
                else:
                    global ham_emails
                    ham_emails += 1
                    train_emails[f.read()] = 0
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


def calculate_total_number_of_words():
    global total_number_of_ham_words, total_number_of_spam_words
    for word in trained_words:
        total_number_of_spam_words += trained_words[word][0]  # spam
        total_number_of_ham_words += trained_words[word][1]  # ham


def compute_conditional_probs():
    for word in trained_words:
        trained_words[word][0] /= total_number_of_spam_words
        trained_words[word][1] /= total_number_of_ham_words


def classify_email(email):
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


def calculate_accuracy():
    correct = 0
    for email in test_emails:
        if classify_email(email) == test_emails[email]:
            correct += 1
    return correct / len(test_emails)


# training
path = "input/lemm_stop/"
read_training_emails(path + "train")

# sort()
apply_laplace()
calculate_total_number_of_words()

p_ham = ham_emails / (ham_emails + spam_emails)
p_spam = spam_emails / (ham_emails + spam_emails)
compute_conditional_probs()

print_in_wordsc(str(trained_words))

# testing
read_test_emails(path + "test")

print(f"Accuracy: {calculate_accuracy() * 100:.2f}")


def cross_leave_one_out(path):
    correct = 0
    trained_words = {}
    test_emails = {}
    train_emails = {}
    spam_emails = 0
    ham_emails = 0

    total_number_of_ham_words = 0
    total_number_of_spam_words = 0
    p_spam = 0
    p_ham = 0

    for email_to_skip in train_emails.keys():
        for email in train_emails.keys():
            if email != email_to_skip:
                if classify_email(email) == train_emails[email]:
                    correct += 1
