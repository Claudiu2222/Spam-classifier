import os
from collections import Counter
import numpy as np


def read_training_emails(path):
    emails = []
    labels = []
    for folder in os.listdir(path):
        if folder == ".DS_Store" or folder == "part10":
            continue
        for filename in os.listdir(os.path.join(path, folder)):
            if filename.endswith("txt"):
                file_path = os.path.join(path, folder, filename)
                with open(file_path) as f:
                    emails.append(f.read())
                    labels.append(1 if "spm" in filename else 0)
    return emails, labels


def read_test_emails(path):
    emails = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith("txt"):
            file_path = os.path.join(path, filename)
            with open(file_path) as f:
                emails.append(f.read())
                labels.append(1 if "spm" in filename else 0)
    return emails, labels


mails, labels = read_training_emails("input/bare")
test_mails, test_labels = read_test_emails("input/bare/part10")
# print("Total emails:", len(mails))
# print("Spam emails:", len([label for label in labels if label == 1]))

total_emails = len(mails)
total_spam_emails = len([label for label in labels if label == 1])


def tokenize(text):
    return text.split(" ")


def extract_vocabulary_and_word_counts(emails, labels):
    vocabulary = set()
    count_word_is_spam = Counter()
    count_word_is_not_spam = Counter()
    for index, email in enumerate(emails):
        vocabulary.update(tokenize(email))
        if labels[index] == 1:
            for token in tokenize(email):
                count_word_is_spam[token] += 1
        else:
            for token in tokenize(email):
                count_word_is_not_spam[token] += 1
    return vocabulary, count_word_is_spam, count_word_is_not_spam


# def remove_less_frequent_words():
#     new_vocab = set()
#     for token in vocabulary:
#         if count_word_is_spam[token] + count_word_is_not_spam[token] >= 2:
#             new_vocab.add(token)
#     return new_vocab


vocabulary, count_word_is_spam, count_word_is_not_spam = extract_vocabulary_and_word_counts(
    mails, labels
)

vocabulary_positions = dict(zip(vocabulary, range(len(vocabulary))))


def vectorize_mail(email):
    vector = np.zeros(len(vocabulary))
    for token in tokenize(email):
        if token in vocabulary:
            vector[vocabulary_positions[token]] = 1
    return vector


def vectorize_mails(emails):
    vectors = []
    for email in emails:
        vectors.append(vectorize_mail(email))
    return np.array(vectors)


vectors = vectorize_mails(mails)


def calculate_conditional_probabilities():
    # prima linie e pt not spam, a doua pt spam
    conditional_probabilities = np.zeros((2, len(vocabulary)))
    for token in vocabulary:
        conditional_probabilities[0, vocabulary_positions[token]] = (
            count_word_is_not_spam[token] + 1
        ) / (total_emails - total_spam_emails + len(vocabulary))
        conditional_probabilities[1, vocabulary_positions[token]] = (
            count_word_is_spam[token] + 1
        ) / (total_spam_emails + len(vocabulary))
    return conditional_probabilities


def calculate_class_probabilities():
    p_spam = total_spam_emails / total_emails
    p_not_spam = (total_emails - total_spam_emails) / total_emails
    return p_not_spam, p_spam


p_spam, p_not_spam = calculate_class_probabilities()
conditional_probabilities = calculate_conditional_probabilities()


def classify_1(email):
    vector_current_email = vectorize_mail(email)

    p_spam_given_email = np.log(p_spam)
    p_not_spam_given_email = np.log(p_not_spam)
    for index, token in enumerate(vocabulary):
        if vector_current_email[vocabulary_positions[token]] == 1:
            p_spam_given_email += np.log(conditional_probabilities[1, index])
            p_not_spam_given_email += np.log(
                conditional_probabilities[0, index])
        # else:
        #     p_spam_given_email += np.log(1 -
        #                                  conditional_probabilities[1, index])
        #     p_not_spam_given_email += np.log(1 -
        #                                      conditional_probabilities[0, index])
    return 1 if p_spam_given_email > p_not_spam_given_email else 0


def calculate_accuracy():
    correct = 0
    for index, email in enumerate(test_mails):
        if classify_1(email) == test_labels[index]:
            correct += 1
    return correct / len(test_mails)


print("Accuracy:", calculate_accuracy())
