import csv
import os
import sys

train_labeled_studies = "./MURA-v1.1/train_labeled_studies.csv"
valid_labeled_studies = "./MURA-v1.1/valid_labeled_studies.csv"

def is_image(filename):
	#https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st
	data = open(filename,'rb').read(10)

	# check if file is PNG
	if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
		return True

	return False




def init_dirs():
	# Clean
	os.system("rm -rf Dataset")
	# Directory structure
	os.system("mkdir Dataset")
	os.system("mkdir Dataset/train")
	os.system("mkdir Dataset/valid")
	os.system("mkdir Dataset/train/negative")
	os.system("mkdir Dataset/train/positive")
	os.system("mkdir Dataset/valid/negative")
	os.system("mkdir Dataset/valid/positive")

def copy_files(csvfile, name):
	i=0
	cwd = os.getcwd()
	with open(csvfile) as stud_label:
		csvFile = csv.reader(stud_label, delimiter=',')
		for row in csvFile:
			if row[1] == '1':
				loc = "./Dataset/"+name+"/positive/"
			else:
				loc = "./Dataset/"+name+"/negative/"

			new_name = row[0]
			new_name = new_name.replace("MURA-v1.1/"+name,'')
			new_name = new_name.replace('/','')

			images_in_file = os.listdir(row[0])

			for imgf in images_in_file:

				if is_image(row[0]+imgf):
					os.system("ln "+row[0]+imgf+" "+loc+new_name+imgf)
					i +=1
					if i %500 == 0:
						print(i)
				else:
					print("Skipping non-png.")



if __name__ == "__main__":
	if len(sys.argv) == 2:
		if sys.argv[1] == "init":
			init_dirs()

	copy_files(train_labeled_studies,'train')
	copy_files(valid_labeled_studies,'valid')


