import os
from collections import Counter
import numpy as np


def read_training_emails(path):
    emails = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith("txt"):
            file_path = os.path.join(path, filename)
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


def extract_mose_relevant_words():
    new_vocabulary = set()
    new_vocabulary = sorted(
        vocabulary, key=lambda word: count_word_is_not_spam[word] + count_word_is_spam[word], reverse=True)[:2500]

    # for word in new_vocabulary:
    #     if len(word) < 4:
    #         new_vocabulary.remove(word)

    # new_vocabulary = new_vocabulary[:-1]

    return new_vocabulary


mails, labels = read_training_emails("input/bare/train")
test_mails, test_labels = read_test_emails("input/bare/test")
# print("Total emails:", len(mails))
# print("Spam emails:", len([label for label in labels if label == 1]))

total_emails = len(mails)
total_spam_emails = len([label for label in labels if label == 1])


def tokenize(text):
    return text.lower().split(" ")


def extract_vocabulary_and_word_counts(emails, labels):
    vocabulary = set()
    count_word_is_spam = Counter()
    count_word_is_not_spam = Counter()
    for index, email in enumerate(emails):
        vocabulary.update(tokenize(email))
        if labels[index] == 1:
            for token in tokenize(email):
                if token.isdigit():
                    token = "number"
                count_word_is_spam[token] += 1
        else:
            for token in tokenize(email):
                if token.isdigit():
                    token = "number"
                count_word_is_not_spam[token] += 1
    return vocabulary, count_word_is_spam, count_word_is_not_spam


vocabulary, count_word_is_spam, count_word_is_not_spam = extract_vocabulary_and_word_counts(
    mails, labels
)

vocabulary = extract_mose_relevant_words()

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
total_spam_words = sum(count_word_is_spam.values())
total_not_spam_words = sum(count_word_is_not_spam.values())


def calculate_conditional_probabilities(vocabulary):
    # prima linie e pt not spam, a doua pt spam
    conditional_probabilities = np.zeros((2, len(vocabulary)))
    for token in vocabulary:
        conditional_probabilities[0, vocabulary_positions[token]] = (
            count_word_is_not_spam[token] + 1
        ) / (total_not_spam_words + len(vocabulary))
        conditional_probabilities[1, vocabulary_positions[token]] = (
            count_word_is_spam[token] + 1
        ) / (total_spam_words + len(vocabulary))
    return conditional_probabilities


def calculate_class_probabilities():
    p_spam = total_spam_emails / total_emails
    p_not_spam = (total_emails - total_spam_emails) / total_emails
    return p_not_spam, p_spam


# def extract_most_important_words(email):
#     # sort the word by their frequency in the count_word_is_spam and count_word_is_not_spam
#     # and return the first 50 words
#     words = tokenize(email)
#     return " ".join([word for word in sorted(words, key=lambda word: count_word_is_spam[word] + count_word_is_not_spam[word], reverse=True)[:min(50, len(words))]])


p_spam, p_not_spam = calculate_class_probabilities()
conditional_probabilities = calculate_conditional_probabilities(vocabulary)


def classify_1(email, conditional_probabilities=conditional_probabilities):
    vector_current_email = vectorize_mail(email)
    p_spam_given_email = np.log(p_spam)
    p_not_spam_given_email = np.log(p_not_spam)
    for index, token in enumerate(vocabulary):
        if vector_current_email[vocabulary_positions[token]] == 1:
            p_spam_given_email += np.log(conditional_probabilities[1, index])
            p_not_spam_given_email += np.log(
                conditional_probabilities[0, index])
        else:
            p_spam_given_email += np.log(abs(1 -
                                         conditional_probabilities[1, index]))
            p_not_spam_given_email += np.log(abs(1 -
                                             conditional_probabilities[0, index]))
    return 1 if p_spam_given_email > p_not_spam_given_email else 0

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
    for index, email in enumerate(test_mails):
        if classify_1(email) == test_labels[index]:
            correct += 1
    return correct / len(test_mails)


print("Accuracy:", calculate_accuracy() * 100)


def print_to_file_sorted_words_and_appearence_count():
    with open("wordsc.txt", "w") as f:
        for word in sorted(vocabulary, key=lambda word: count_word_is_not_spam[word] + count_word_is_spam[word], reverse=True):
            f.write(
                f"{word} {count_word_is_spam[word] + count_word_is_not_spam[word]}\n")


print_to_file_sorted_words_and_appearence_count()


def print_conditional_probabilities():
    with open("condprob.txt", "w") as f:
        for index, token in enumerate(vocabulary):
            f.write(
                f"{token} {conditional_probabilities[0, index]} {conditional_probabilities[1, index]}\n")


print_conditional_probabilities()

print(count_word_is_not_spam["."], count_word_is_spam["."])