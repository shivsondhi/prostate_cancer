'''
ML techniques on the features - CALLABLE FUNCTIONS ONLY.
'''

# import statements
import warnings
import pandas as pd
import numpy as np
import sklearn as sk
from pprint import pprint
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# set import variables
pd.set_option("expand_frame_repr", False)
pd.set_option("max_columns", 9)
np.random.seed(0)
warnings.filterwarnings("ignore")


def do_t_stage(t_data, filenames, mode):
	if mode == 'partial':
		print(t_data[0].head())
	# FEATURE SELECTION
	# Select K-Best Features, Scale Values and Select from RF Model
	kbest = SelectKBest(score_func=chi2, k=10000)
	scaler = StandardScaler()
	fs_data = []
	for i, d in enumerate(t_data):
		print("\nFILENAME: {}".format(filenames[i]))
		t_rows = list(d.index)
		t_columns = d.columns[:-3]
		# K-best
		selector = kbest.fit(d.iloc[:, :-3], d.iloc[:, -3])
		t_columns = t_columns[selector.get_support()]
		fs_data.append(pd.DataFrame(selector.transform(t_data[i].iloc[:, :-3]), columns = t_columns, index=t_rows))
		if mode == 'show':
			print("Selecting k best features -\n", fs_data[i].head())
		# Scale 
		t_columns = fs_data[i].columns
		fs_data[i] = pd.DataFrame(scaler.fit_transform(fs_data[i]), columns=t_columns, index=t_rows)
		if mode == 'show':
			print("Scaling data -\n", fs_data[i].head())
		# Select from RF
		classifier = RandomForestClassifier(n_estimators=1)
		classifier = classifier.fit(fs_data[i], d['TStage'])
		selector = SelectFromModel(classifier, prefit=True)
		t_columns = t_columns[selector.get_support()]
		fs_data[i] = pd.DataFrame(selector.transform(fs_data[i]), columns=t_columns, index=t_rows)
		fs_data[i]['TStage'] = d['TStage']
		if mode in ('partial', 'show'):
			print("Selecting data from RF model -\n", fs_data[i].head())
		print("Shape after feature selection: {}".format(fs_data[i].shape), end="\n\n")
	# RESAMPLING the data - SMOTEENN
	balanced_data = [[] for _ in range(2)]
	for i, d in enumerate(fs_data):
		sme = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=3))
		x, y = sme.fit_resample(fs_data[i], t_data[i]['TStage'])
		# x are the features and y are the targets
		balanced_data[i].append(x)
		balanced_data[i].append(y)
		if mode == 'show':
			print("FILENAME: {}".format(filenames[i]), Counter(balanced_data[i][1]))
	# DIMENSIONALITY REDUCTION
	# Kernel PCA and LDA (can be toggled on or off)
	pca = True
	pca_dim = 31
	lda = True
	lda_dim = 5
	if pca or lda:
		dr_data = []
		for i in range(len(filenames)):
			print("\nFILENAME: {}".format(filenames[i]))
			if pca:
				decomposer = KernelPCA(n_components=pca_dim, kernel='rbf', gamma=0.05, degree=7)
				dr_data.append(decomposer.fit_transform(balanced_data[i][0]))
				print("Shape and type after PCA: ", dr_data[i].shape, type(dr_data[i]))
			if lda:
				decomposer = LinearDiscriminantAnalysis(n_components=lda_dim, solver='eigen')
				dr_data[i] = decomposer.fit_transform(dr_data[i], balanced_data[i][1])
				print("Shape and type after LDA: ", dr_data[i].shape, type(dr_data[i]))
	else:
		dr_data.append(balanced_data[0][0])
		dr_data.append(balanced_data[1][0])
	# CLASSIFICATION
	splits = 7
	seed = 7
	kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
	results = {'SVM': [],
				'RF': [],
				'KNN': [],
				'NB': []
				}
	for i, d in enumerate(dr_data):
		# SVM
		res = []
		classifier = SVC(gamma='auto')
		results['SVM'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['SVM'][i] = results['SVM'][i].mean()
		# RF
		classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=7, criterion='gini')
		results['RF'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['RF'][i] = results['RF'][i].mean()
		# KNN
		knn = KNeighborsClassifier(n_neighbors=5)
		results['KNN'].append(cross_val_score(knn, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['KNN'][i] = results['KNN'][i].mean()
		# NB
		nb = GaussianNB()
		results['NB'].append(cross_val_score(nb, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['NB'][i] = results['NB'][i].mean()
	print("\nFinal Results for datasets: {0}, {1} -".format(filenames[0], filenames[1]))
	pprint(results)
	# PLOTTING
	# PCA
	pca = PCA(n_components = 3)
	x_pca = pca.fit_transform(balanced_data[0][0])
	fig = plt.figure(figsize=(10, 6))
	plt.suptitle("3-D plot for resampled data using dimesnionality reduction (T-Stage)\n\n")
	ax = fig.add_subplot(121, projection='3d')
	ax.set_title("PCA\n\n")
	ax.view_init(elev=15,azim=66)
	for i in range(len(balanced_data[0][1])):
		if balanced_data[0][1][i] == 0:
			t2a = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='y', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 1:
			t2b = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='g', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 2:
			t2c = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='b', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 3:
			t3a = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='r', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 4:
			t3b = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='m', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 5:
			t4 = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='c', label=balanced_data[0][1][i])
	plt.legend((t2a, t2b, t2c, t3a, t3b, t4),
		('T2a', 'T2b', 'T2c','T3a','T3b', 'T4'),
		scatterpoints=1,
		loc='upper right',
		ncol=1,
		fontsize=10)
	# PCA + LDA
	pca = PCA(n_components = 10)
	x_pca = pca.fit_transform(balanced_data[0][0])
	lda = LinearDiscriminantAnalysis(n_components = 3)
	x_lda = lda.fit_transform(x_pca, balanced_data[0][1])
	ax = fig.add_subplot(122, projection='3d')
	plt.title("PCA & LDA\n\n")
	ax.view_init(elev=-79,azim=-7)
	for i in range(len(balanced_data[0][1])):
		if balanced_data[0][1][i] == 0:
			t2a = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='y', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 1:
			t2b = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='g', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 2:
			t2c = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='b', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 3:
			t3a = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='r', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 4:
			t3b = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='m', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 5:
			t4 = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='c', label=balanced_data[0][1][i])
	plt.legend((t2a, t2b, t2c, t3a, t3b, t4),
				('T2a', 'T2b', 'T2c','T3a','T3b', 'T4'),
				scatterpoints=1,
				loc='upper right',
				ncol=1,
				fontsize=10)
	#plt.show()
	return results


def do_gleason(t_data, filenames, mode):
	# FEATURE SELECTION
	# Select K-Best Features, Scale, VarianceThreshold and Select From RF Model
	kbest = SelectKBest(score_func=chi2, k=15000)
	scaler = StandardScaler()
	thresholding = VarianceThreshold()
	fs_data = []
	for i, d in enumerate(t_data):
		print("\nFILENAME: {}".format(filenames[i]))
		t_rows = list(d.index)
		t_columns = d.columns[:-3]
		# K-best
		selector = kbest.fit(d.iloc[:, :-3], d.iloc[:, -2])
		t_columns = t_columns[selector.get_support()]
		fs_data.append(pd.DataFrame(selector.transform(t_data[i].iloc[:, :-3]), columns = t_columns, index=t_rows))
		if mode == 'show':
			print("Selecting k best features -\n", fs_data[i].head())
		# Scale 
		t_columns = fs_data[i].columns
		fs_data[i] = pd.DataFrame(scaler.fit_transform(fs_data[i]), columns=t_columns, index=t_rows)
		if mode == 'show':
			print("Scaling data -\n", fs_data[i].head())
		# Variance Threshold
		fs_data[i] = pd.DataFrame(thresholding.fit_transform(fs_data[i]), columns=t_columns, index=t_rows)
		if mode == 'show':
			print("After variance thresholding -\n", fs_data[i].head())
		# Select from RF
		classifier = RandomForestClassifier(n_estimators=1)
		classifier = classifier.fit(fs_data[i], d['Gleason'])
		selector = SelectFromModel(classifier, prefit=True)
		t_columns = t_columns[selector.get_support()]
		fs_data[i] = pd.DataFrame(selector.transform(fs_data[i]), columns=t_columns, index=t_rows)
		fs_data[i]['Gleason'] = d['Gleason']
		if mode in ('show'):
			print("Selecting data from RF model -\n", fs_data[i].head())
		print("Shape after feature selection: {}".format(fs_data[i].shape), end="\n\n")
	# RESAMPLING data - SMOTEENN
	balanced_data = [[] for _ in range(2)]
	for i, d in enumerate(fs_data):
		sme = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=1))
		x, y = sme.fit_resample(fs_data[i], t_data[i]['Gleason'])
		# x are the features and y are the targets
		balanced_data[i].append(x)
		balanced_data[i].append(y)
		if mode == 'show':
			print("FILENAME: {}".format(filenames[i]), Counter(balanced_data[i][1]))
	# DIMENSIONALITY REDUCTION
	# Kernel PCA and LDA (can be toggled on or off)
	pca = False
	pca_dim = 31
	lda = True
	lda_dim = 3
	if pca or lda:
		dr_data = []
		for i in range(len(filenames)):
			print("\nFILENAME: {}".format(filenames[i]))
			if pca:
				decomposer = KernelPCA(n_components=pca_dim, kernel='rbf', gamma=0.05, degree=7)
				dr_data.append(decomposer.fit_transform(balanced_data[i][0]))
				print("Shape and type after PCA: ", dr_data[i].shape, type(dr_data[i]))
			else:
				dr_data.append(balanced_data[i][0])
			if lda:
				decomposer = LinearDiscriminantAnalysis(n_components=lda_dim)
				dr_data[i] = decomposer.fit_transform(dr_data[i], balanced_data[i][1])
				print("Shape and type after LDA: ", dr_data[i].shape, type(dr_data[i]))
	else:
		dr_data.append(balanced_data[0][0])
		dr_data.append(balanced_data[1][0])
	# CLASSIFICATION
	splits = 10
	seed = 7
	kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
	results = {'SVM': [],
				'RF': [],
				'KNN': [],
				'NB': []
				}
	for i, d in enumerate(dr_data):
		# SVM
		res = []
		classifier = SVC(gamma='auto')
		results['SVM'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['SVM'][i] = results['SVM'][i].mean()
		# RF
		# rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=10,max_features='auto')
		classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=7, max_features='auto', criterion='gini') #, n_jobs=-1
		results['RF'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['RF'][i] = results['RF'][i].mean()
		# KNN
		k_scores = []
		for n in range(1, 16):
			knn = KNeighborsClassifier(n_neighbors=3)
			scores = (cross_val_score(knn, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
			k_scores.append(scores.mean())
		results['KNN'].append(max(k_scores))
		# NB
		nb = GaussianNB()
		results['NB'].append(cross_val_score(nb, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['NB'][i] = results['NB'][i].mean()
	print("\nFinal Results for datasets: {0}, {1} -".format(filenames[0], filenames[1]))
	pprint(results)
	# PLOTTING
	# PCA
	pca = PCA(n_components = 3)
	x_pca = pca.fit_transform(balanced_data[0][0])
	fig = plt.figure(figsize=(13, 7))
	plt.suptitle("3-D plot for resampled data using dimesnionality reduction (Gleason Score)\n\n")
	ax = fig.add_subplot(121, projection='3d')
	ax.set_title("PCA\n\n")
	ax.view_init(elev=177,azim=-96)
	for i in range(len(balanced_data[0][1])):
		if balanced_data[0][1][i] == 6:
			six = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='y', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 7:
			seven = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='g', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 8:
			eight = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='b', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 9:
			nine = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='r', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 10:
			ten = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='m', label=balanced_data[0][1][i])
	plt.legend((six, seven, eight, nine, ten),
		('6', '7', '8','9','10'),
		scatterpoints=1,
		loc='upper right',
		ncol=1,
		fontsize=10)
	# PCA + LDA
	pca = PCA(n_components = 10)
	x_pca = pca.fit_transform(balanced_data[0][0])
	lda = LinearDiscriminantAnalysis(n_components = 3)
	x_lda = lda.fit_transform(x_pca, balanced_data[0][1])
	ax = fig.add_subplot(122, projection='3d')
	plt.title("PCA & LDA\n\n")
	ax.view_init(elev=10,azim=-112)
	for i in range(len(balanced_data[0][1])):
		if balanced_data[0][1][i] == 6:
			six = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='y', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 7:
			seven = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='g', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 8:
			eight = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='b', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 9:
			nine = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='r', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 10:
			ten = ax.scatter(x_lda[i][0], x_lda[i][1], x_lda[i][2], c='m', label=balanced_data[0][1][i])
	plt.legend((six, seven, eight, nine, ten),
				('6', '7', '8','9','10'),
				scatterpoints=1,
				loc='upper right',
				ncol=1,
				fontsize=10)
	#plt.show()
	return results


def do_t_recur(t_data, filenames, mode):
	# FEATURE SELECTION
	# Scale, Use VarianceThreshold and Pearson Correlation, and Select From RF Model
	scaler = MinMaxScaler()
	thresholding = VarianceThreshold()
	fs_data = []
	for i, d in enumerate(t_data):
		print("\nFILENAME: {}".format(filenames[i]))
		t_rows = list(d.index)
		t_columns = d.columns[:-3]
		# Replace NaN values with the column mean
		t_data[i]['Recurrence'].fillna((t_data[i]['Recurrence'].mean()), inplace=True)
		# Scale
		fs_data.append(pd.DataFrame(scaler.fit_transform(t_data[i].iloc[:, :-3]), columns=t_columns, index=t_rows))
		if mode == 'show':
			print("Scaling data -\n", fs_data[i].head())
		# Variance Threshold
		selector = thresholding.fit(fs_data[i])
		t_columns = t_columns[thresholding.get_support()]
		fs_data[i] = pd.DataFrame(thresholding.transform(fs_data[i]), columns=t_columns, index=t_rows)
		if mode == 'show':
			print("After variance thresholding -\n", fs_data[i].head())
		# Select From RF
		classifier = RandomForestClassifier(n_estimators=1)
		classifier = classifier.fit(fs_data[i], d['Recurrence'])
		selector = SelectFromModel(classifier, prefit=True)
		t_columns = t_columns[selector.get_support()]
		fs_data[i] = pd.DataFrame(selector.transform(fs_data[i]), columns=t_columns, index=t_rows)
		fs_data[i]['Recurrence'] = d['Recurrence']
		if mode in ('show'):
			print("Selecting data from RF model -\n", fs_data[i].head())
		print("Shape after feature selection: {}".format(fs_data[i].shape), end="\n\n")
	# RESAMPLING data - SMOTEENN
	balanced_data = [[] for _ in range(2)]
	for i, d in enumerate(fs_data):
		sme = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=2))
		x, y = sme.fit_resample(fs_data[i], t_data[i]['Recurrence'])
		# x are the features and y are the targets
		balanced_data[i].append(x)
		balanced_data[i].append(y)
		print("Upsampling the data... in {}".format(filenames[i]))
		if mode == 'show':
			print("FILENAME: {}".format(filenames[i]), Counter(balanced_data[i][1]))
	# DIMENSIONALITY REDUCTION
	# Kernel PCA (can be toggled on or off)
	pca = True
	pca_dim = 20
	if pca:
		dr_data = []
		for i in range(len(filenames)):
			print("\nFILENAME: {}".format(filenames[i]))
			decomposer = KernelPCA(n_components=pca_dim, kernel='rbf', gamma=0.5, degree=7)
			dr_data.append(decomposer.fit_transform(balanced_data[i][0]))
			print("Shape and type after PCA: ", dr_data[i].shape, type(dr_data[i]))
	else:
		dr_data.append(balanced_data[0][0])
		dr_data.append(balanced_data[1][0])
	# CLASSIFICATION
	splits = 10
	seed = 7
	kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
	results = {'SVM': [],
				'RF': [],
				'KNN': [],
				'NB': []
				}
	for i, d in enumerate(dr_data):
		# SVM
		res = []
		classifier = SVC(gamma='auto')
		results['SVM'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['SVM'][i] = results['SVM'][i].mean()
		# RF
		# rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=10,max_features='auto')
		classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=7, max_features='auto', criterion='gini', n_jobs=-1)
		results['RF'].append(cross_val_score(classifier, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['RF'][i] = results['RF'][i].mean()
		# KNN
		k_scores = []
		for n in range(1, 16):
			knn = KNeighborsClassifier(n_neighbors=3)
			scores = (cross_val_score(knn, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
			k_scores.append(scores.mean())
		results['KNN'].append(max(k_scores))
		# NB
		nb = GaussianNB()
		results['NB'].append(cross_val_score(nb, pd.DataFrame(dr_data[i]), balanced_data[i][1], cv=kfold))
		results['NB'][i] = results['NB'][i].mean()
	print("\nFinal Results for datasets: {0}, {1} -".format(filenames[0], filenames[1]))
	pprint(results)
	# PLOTTING
	# PCA
	pca = PCA(n_components = 3)
	x_pca = pca.fit_transform(balanced_data[0][0])
	fig = plt.figure(figsize=(13, 7))
	plt.suptitle("3-D plot for resampled data using dimesnionality reduction (Biomedical Recurrence)\n\n")
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("PCA\n\n")
	ax.view_init(elev=177,azim=-96)
	for i in range(len(balanced_data[0][1])):
		if balanced_data[0][1][i] == 0:
			false = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='y', label=balanced_data[0][1][i])
		elif balanced_data[0][1][i] == 1:
			true = ax.scatter(x_pca[i][0], x_pca[i][1], x_pca[i][2], c='g', label=balanced_data[0][1][i])
	plt.legend((false, true),
		("Didn't recur", "Recurred"),
		scatterpoints=1,
		loc='upper right',
		ncol=1,
		fontsize=10)
	#plt.show()
	return results


def main():
	print('This file contains only callable functions. Run "final_compiled.py" instead.')


if __name__ == "__main__":
	main()