import sys
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

TRAIN_DATA = "spambase/train-data.csv"
TRAIN_TARGETS= "spambase/train-targets.csv"
TEST_DATA = "spambase/test-data.csv"

if __name__ == '__main__':

	with open(TRAIN_DATA) as f:
		X_train = np.loadtxt(f, delimiter=',')
	f.close()

	with open(TRAIN_TARGETS) as f:
		y_train = np.loadtxt(f, dtype='int_')
	f.close()

	with open(TEST_DATA) as f:
		X_test = np.loadtxt(f, delimiter=',')
	f.close()

	print("Dataset stored")

	# 5-fold cross-validation
	# random_state ensures same split for each value of gamma
	kf = KFold(n_splits=5, shuffle=True, random_state=42)

	gamma_values = [0.05, 0.01, 0.005, 0.001, 0.0005]

	accuracy_scores = []

	scoring = ['accuracy','precision', 'recall', 'f1']

	# Do model selection over all the possible values of gamma 
	for gamma in gamma_values:

		# Train a classifier with current gamma
		clf = SVC(C=10, kernel='rbf', gamma=gamma)
		print("Classifier trained")
		# Compute cross-validated accuracy scores
		scores = cross_validate(clf, X_train, y_train, cv=kf.split(X_train), scoring=scoring, n_jobs=-1)
		
		# Compute the mean accuracy and print it
		accuracy_score = scores['test_accuracy'].mean()
		print('Mean accuracy with gamma={}: {}'.format(gamma, accuracy_score))
		# Compute the mean precision and print it
		precision_score = scores['test_precision'].mean()
		print('Mean precision with gamma={}: {}'.format(gamma, precision_score))
		# Compute the mean recall and print it
		recall_score = scores['test_recall'].mean()
		print('Mean recall with gamma={}: {}'.format(gamma, recall_score))
		# Compute the mean f1 and print it
		f1_score = scores['test_f1'].mean()
		print('Mean F1 with gamma={}: {}'.format(gamma, f1_score))

		# Keep track of the mean accuracy
		accuracy_scores.append(accuracy_score)

	# Get the gamma with highest mean accuracy
	best_index = np.array(accuracy_scores).argmax()
	best_gamma = gamma_values[best_index]

	plt.figure()
	plt.title("Learning curves (SVM, RBF kernel, gamma={})".format(best_gamma))
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.grid()

	clf = SVC(C=10, kernel='rbf', gamma=best_gamma)

	# Compute the scores of the learning curve
	# by default the (relative) dataset sizes are: 10%, 32.5%, 55%, 77.5%, 100% 
	train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5)

	# Get the mean and std of train and test scores along the varying dataset sizes
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	# Plot the mean and std for the training scores
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")

	# Plot the mean and std for the cross-validation scores
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

	plt.legend()
	plt.savefig('plot.png')
	plt.show()
	

	# Predict label for the train set
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	np.savetxt('test-labels.txt', y_pred, delimiter=',', fmt='%i')
