#Libraries, numpy, sklearn/scikit-learn




import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

Features = [
    ("Parent", "Eat your vegetables."),
    ("Child", "I wanna play games"),
    ("Parent", "Do your homework."),
    ("Child", "Let's go play."),
    ("Parent", "I'm busy."),
    ("Child", "You're to slow."),
    ("Parent", "Don't drink so much."),
    ("Child", "This is so boring."),
    ("Parent", "Stay focused."),
    ("Child", "Can I go now."),
    ("Parent", "I need to make some money."),
    ("Child", "I hate school."),
    ("Parent", "One day you'll understand."),
    ("Child", "I dont wanna go to sleep."),
    ("Parent", "I'm so tired."),
    ("Child", "Do I have to."),
    ("Parent", "No arguing."),
    ("Child", "I'm so hungry."),
    # Add more examples as needed
]

# Split data into features (X) and labels (y)
y, X = zip(*Features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Convert the text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the classifier
report = classification_report(y_test, y_pred, zero_division=1)
print(y_test, "\n",y_pred)




print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)


#Inputs to prediction
"""
inputToBeEvaluated = input("What should be classified: ")
outputEvaluated = classifier.predict(inputToBeEvaluated)
print(f"Classified as: {outputEvaluated}")
"""