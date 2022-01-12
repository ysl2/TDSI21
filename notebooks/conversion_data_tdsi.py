from shutil import copy2, copyfileobj
import glob
import gzip
import os
import sys
import re
import argparse
import json

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser(description='Convertisseur dataset')
parser.add_argument('-i','--inputFolder', help='Input data folder', required=True)
parser.add_argument('-t','--taskNumber', help='Task number', required=False, default="501") ##must be > 500
parser.add_argument('-f','--dataOrganization', help='Data organization file (test and train)', required=True)

args = parser.parse_args()


from nnunet.dataset_conversion.utils import generate_dataset_json

# original data : /scratch/tgreni02/Projet/20_Epaule_Pialat/Data/mDixon_e8
original_folder = args.inputFolder + "/" #'/local/cnicol03/Segmentation_epaule/mDixon_e8/'
dest_folder = os.environ['nnUNet_raw_data_base']+"/nnUNet_raw_data/Task"+args.taskNumber + "_Epaule/"


def createFolderStruct():
	testData= []
	trainData = []
	#load data organization from dataOrganization file
	with open(args.dataOrganization) as json_file:
		data = json.load(json_file)
		for te in data['test']:
			testData.append(te)

		for tr in data['training']:
			trainData.append(tr)
		
	#copy test data and labels in test folder 
	for folder in testData:
		source_folder = original_folder+"P"+str(folder)
		copy_nifti(source_folder, "imagesTs")

		copy_labels(source_folder, "labelTs")
		compress_labels("labelTs")



	#copy train data and labels in train folder 
	for folder in trainData:
		source_folder = original_folder+"P"+str(folder)
		copy_nifti(source_folder, "imagesTr")

		copy_labels(source_folder, "labelTr")
		compress_labels("labelTr")

	
	#normalize the name of image file
	rename_files("imagesTs")
	rename_files("imagesTr")
	
	#normalize the name of the label file
	rename_label("labelTr")
	rename_label("labelTs")

	#create the json file as a description of the dataset
	create_json()

	
	

	
def create_json():
	print("\ncreating json\n")
	dataset = dest_folder + "dataset.json"
	imagesTr = dest_folder + "imagesTr"
	imagesTs = dest_folder + "imagesTs"
	modalities = ("MRI_Dixon",)
	labels = {0: "background", 1: "Suscapullaire", 2: "Supraepineux", 3: "Infra-epineux", 4: "Petit-rond", 5: "Deltoide"}
	name = "Epaule"
	generate_dataset_json(dataset, imagesTr, imagesTs, modalities, labels, name)

def compress_labels(destSubFolder):
	print("\n\ncompressing label files")
	to_compress_files = glob.iglob(dest_folder + destSubFolder + "/*.nii")
	for uncompressed_file in to_compress_files:
		with open(uncompressed_file, 'rb') as f_in:
			with gzip.open(uncompressed_file + ".gz", 'wb') as f_out:
				copyfileobj(f_in, f_out)
		os.remove(uncompressed_file)

def copy_nifti(sourceFolder, destSubFolder):
	if not os.path.exists(dest_folder + destSubFolder):
		os.makedirs(dest_folder + destSubFolder)
	
	nifti_files = glob.iglob(sourceFolder + "/*.nii.gz")
	for nifti_file in nifti_files:
		print("copying file: " + nifti_file + "  --->  " + dest_folder + destSubFolder)
		copy2(nifti_file, dest_folder + destSubFolder)

def copy_labels(sourceFolder, desSubFolder):
		label_files = glob.iglob(sourceFolder + "/*ManualSegmentation.nii")
		if not os.path.exists(dest_folder + desSubFolder):
			os.makedirs(dest_folder + desSubFolder)
	
		for label_file in label_files:
			copy2(label_file, dest_folder + desSubFolder)

def rename_files(subFolder): 
	for f in glob.iglob(dest_folder + subFolder + '/*.nii.gz'):
		search = re.search('P(\d*)', f)
		if search != None:
			number = search.group(1)
			new_name = dest_folder + subFolder + "/epaule_"+number.zfill(3) + "_0000.nii.gz"
			os.rename(f, new_name)

def rename_label(subFolder): 
	for f in glob.iglob(dest_folder + subFolder + '/*.nii.gz'):
		search = re.search('P(\d*)', f)
		if search != None:
			number = search.group(1)
			new_name = dest_folder + subFolder + "/epaule_"+number.zfill(3) + ".nii.gz"
			os.rename(f, new_name)



	


createFolderStruct()
print("\n\n---------Done---------")
