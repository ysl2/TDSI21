from shutil import copy2, copyfileobj
import glob
import gzip
import os
import sys
import re
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

sys.path.insert(1, '/local/cnicol03/nnUNet/nnunet/dataset_conversion/')

from nnunet.dataset_conversion.utils import generate_dataset_json

# original data : /scratch/tgreni02/Projet/20_Epaule_Pialat/Data/mDixon_e8
base_folder = '/local/cnicol03/Segmentation_epaule/mDixon_e8/'
dest_folder = "/local/cnicol03/Segmentation_epaule/Task01_Epaule/"
# this folder should have the training and testing subfolders


def createFolderStruct(status):
	if status == "test":
		test_patients_file = "TestingPatients_DL"
		dest_subfolder_images = "imagesTs"
		dest_subfolder_labels = "labelsTs"

	elif status == "train":
		test_patients_file = "TrainingPatients_DL_set0"
		dest_subfolder_images = "imagesTr"
		dest_subfolder_labels = "labelsTr"
	
	elif status == "valid":
		test_patients_file = "ValidPatients_DL_set0"
		dest_subfolder_images = "imagesTr"
		dest_subfolder_labels = "labelsTr"

				
	with open(test_patients_file+'.txt') as f:
		lines = f.readlines()
		
	for folder in lines:
		source_folder = base_folder+folder.strip("\n")
		
		nifti_files = glob.iglob(source_folder + "/*.nii.gz")
		for nifti_file in nifti_files:
			print("copying file: " + nifti_file + "  --->  " + dest_folder + dest_subfolder_images)
			copy2(nifti_file, dest_folder + dest_subfolder_images)
		
		label_files = glob.iglob(source_folder + "/*ManualSegmentation_v2.nii")
		for label_file in label_files:
			print("\ncopying label file: " + label_file + "  --->  " + dest_folder + dest_subfolder_labels+ "\n\n")
			copy2(label_file, dest_folder + dest_subfolder_labels)
	
	
	print("\n\ncompressing label files")
	to_compress_files = glob.iglob(dest_folder + dest_subfolder_labels + "/*.nii")
	for uncompressed_file in to_compress_files:
		with open(uncompressed_file, 'rb') as f_in:
			with gzip.open(uncompressed_file + ".gz", 'wb') as f_out:
				copyfileobj(f_in, f_out)
		os.remove(uncompressed_file)


def create_json(folder):
	print("\ncreating json\n")
	dataset = folder + "dataset.json"
	imagesTr = folder + "imagesTr"
	imagesTs = folder + "imagesTs"
	modalities = ("MRI_Dixon",)
	labels = {0: "background", 1: "Suscapullaire", 2: "Supraepineux", 3: "Infra-epineux", 4: "Petit-rond", 5: "Deltoide"}
	name = "Epaule"

	generate_dataset_json(dataset, imagesTr, imagesTs, modalities, labels, name)

def rename_files(folder): 
	for f in glob.iglob(folder + '/*.nii.gz'):
		number = re.search('P(\d*)', f).group(1)
		new_name = folder + "/epaule_"+number.zfill(3) + ".nii.gz"
		os.rename(f, new_name)


	
#createFolderStruct("test")
#createFolderStruct("train")
#createFolderStruct("valid")


#rename_files(dest_folder + "imagesTr")
#rename_files(dest_folder + "imagesTs")
#rename_files(dest_folder + "labelsTr")
#rename_files(dest_folder + "labelsTs")

#os.system('nnUNet_convert_decathlon_task -i Task01_Epaule -output_task_id 500')
#print("---------Dataset converti pour nnUNet---------")

create_json("/local/cnicol03/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Epaule/")

os.system('nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity')
#os.system('nnUNet_train 3d_fullres nnUNetTrainerV2 500 0')


print("\n\n---------Done---------")
