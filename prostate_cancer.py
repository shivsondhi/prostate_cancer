'''
Predicting Features from the Pathology Report for Prostate Cancer.

Gleason Score
T-Stage
Biomedical Recurrence
'''

# import statements
import os
import warnings
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
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
from sklearn.ensemble import IsolationForest
from feature_processing import do_t_stage, do_gleason, do_t_recur


# set module variables 
pd.set_option("expand_frame_repr", False)
pd.set_option("max_columns", 9)
np.random.seed(0)
warnings.filterwarnings("ignore")


def main():
	modes = ['show', 'partial', 'hide']
	mode = modes[1]

	# filepaths to data downloaded from cBioPortal
	filepath = "C:\\path\\to\\base\\directory\\prad_tcga.tar"			# base directory
	filenames = ["data_methylation_hm450.tsv", "data_RNA_Seq_v2_expression_median.tsv"]
	clinicalfile = "prad_tcga_clinical_data.tsv"\

	# import the gene data (contains gene data to use as x-features while predicting)
	data = []
	for i, filename in enumerate(filenames):
		# data is tab separated
		data.append(pd.read_csv(os.path.join(filepath, filenames[i]), sep="\t"))
		# drop data columns that will not be used and change the index of the DataFrame
		data[i].drop('Entrez_Gene_Id', axis=1, inplace=True)
		data[i].set_index('Hugo_Symbol', inplace=True)
		if mode == 'show':
			print('FILENAME: {0} -\n'.format(filenames[i]), data[i].head())
	# import the clinical data (contains target features to predict)
	clinical_data = pd.read_csv(os.path.join(filepath, clinicalfile), sep="\t")
	clinical_data.set_index('Sample ID', inplace=True)
	if mode == 'show':
		print('FILENAME: clinical_data -\n', clinical_data.head())

	# Transpose the gene data and add the target features to it
	t_data = []
	for i, d in enumerate(data):
		t_data.append(d.fillna(d.mean()).transpose())
		missing = clinical_data.index.difference(d.transpose().index).tolist()
		t_data[i]['TStage'] = clinical_data.drop(missing)['American Joint Committee on Cancer Tumor Stage Code']
		t_data[i]['Gleason'] = clinical_data.drop(missing)['Radical Prostatectomy Gleason Score for Prostate Cancer']
		t_data[i]['Recurrence'] = clinical_data.drop(missing)['Biochemical Recurrence Indicator']
		if mode == 'show':
			print("\n\t\tSkip the missing patient records and append the target values for all the others.\n", "FILENAME: {}".format(filenames[i]))
			print(t_data[i].head())

	# change TStage and Recurrence to integer values
	for i, d in enumerate(t_data):
		t_data[i] = t_data[i][pd.notnull(t_data[i]['TStage'])]
		enc = LabelEncoder()
		# fit encoder to TStage 
		enc.fit(t_data[i]['TStage'].tolist())
		t_data[i]['TStage'] = enc.transform(t_data[i]['TStage'].tolist())
		tstage_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_))
		# fit encoder to Recurrence
		enc.fit(t_data[i]['Recurrence'].tolist())
		t_data[i]['Recurrence'] = enc.transform(t_data[i]['Recurrence'].tolist())
		recurrence_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_))
		if mode == 'show':
			print("\nTStage mapping: ", tstage_mapping)
			print("Recurrence mapping: ", recurrence_mapping)
		# print the DataFrame to check
		if mode == 'show':
			print('\nUpdated dataframe -\n', t_data[i].head())
			print('Value Counts -\n', t_data[i]['TStage'].value_counts())
			print('Value Counts -\n', t_data[i]['Recurrence'].value_counts())
	print("\t\tT-STAGE")
	t_results = t_stage_results = do_t_stage(t_data, filenames, mode)
	print("\n\t\tGLEASON SCORE")
	g_results = gleason_results = do_gleason(t_data, filenames, mode)
	print("\n\t\tTUMOR RECURRENCE")
	r_results = t_recur_results = do_t_recur(t_data, filenames, mode)

	# Plot the final results
	# DNA is the methylation hm450 file
	dna_res = {'SVM': [],
			'RF': [],
			'NB': [],
			'KNN': []
		}
	# RNA is the rna sequencing file
	rna_res = {'SVM': [],
			'RF': [],
			'NB': [],
			'KNN': []
		}
	for i in dna_res.keys():
		dna_res[i].append(g_results[i][0])
		dna_res[i].append(t_results[i][0])
		dna_res[i].append(t_results[i][0])
		rna_res[i].append(g_results[i][1])
		rna_res[i].append(t_results[i][1])
		rna_res[i].append(t_results[i][1])

	fig = plt.figure(figsize=(13, 7))
	fig.suptitle("Final Results")
	# DNA FILE
	ax = fig.add_subplot(121)
	N = len(dna_res['SVM'])
	ind = np.arange(N)
	w = 0.11
	svm = ax.bar(ind, dna_res['SVM'], w, bottom=0, alpha=0.8, color='g')
	rf = ax.bar(ind+w, dna_res['RF'], w, bottom=0, alpha=0.8, color='r')
	nb = ax.bar(ind+2*w, dna_res['NB'], w, bottom=0, alpha=0.8, color='b')
	knn = ax.bar(ind+3*w, dna_res['KNN'], w, bottom=0, alpha=0.8, color='y')
	ax.set_title("Results using the classifiers for different target features on DNA File")
	ax.set_xticks(ind+3*w/2)
	ax.set_xticklabels(('GleasonScore', 'TumorRecurrence', 'T-Stage'))
	ax.legend((svm[0], rf[0], nb[0], knn[0]), ('SVM', 'RF', 'NB', 'KNN'), loc='lower right')
	ax.yaxis.set_units('Percentages (%)')
	# RNA FILE
	ax = fig.add_subplot(122)
	N = len(rna_res['SVM'])
	ind = np.arange(N)
	w = 0.11
	svm = ax.bar(ind, rna_res['SVM'], w, bottom=0, alpha=0.8, color='g')
	rf = ax.bar(ind+w, rna_res['RF'], w, bottom=0, alpha=0.8, color='r')
	nb = ax.bar(ind+2*w, rna_res['NB'], w, bottom=0, alpha=0.8, color='b')
	knn = ax.bar(ind+3*w, rna_res['KNN'], w, bottom=0, alpha=0.8, color='y')
	ax.set_title("Results using the classifiers for different target features on RNA File")
	ax.set_xticks(ind+3*w/2)
	ax.set_xticklabels(('GleasonScore', 'TumorRecurrence', 'T-Stage'))
	ax.legend((svm[0], rf[0], nb[0], knn[0]), ('SVM', 'RF', 'NB', 'KNN'), loc='lower right')
	ax.yaxis.set_units('Percentages (%)')
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()