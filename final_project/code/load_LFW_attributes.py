import os
import cv2
from functools import cmp_to_key

def str_cmp(stra, strb):
	stra = stra[1]
	strb = strb[1]
	if stra > strb:
		return 1
	elif stra == strb:
		return 0
	else:
		return -1

def load_LFW_attributes(filename):

	# read in .txt file
	txt = open(filename, 'r').read()

	# parse by newline
	lines = txt.split("\n")
	identifiers = lines[1].split("\t")

	# sort into dicts
	attributes = []
	for i in range(2, len(lines)):
		elems = lines[i].split("\t")

		new_dict = {}
		new_dict["person"] = elems[0]
		for j in range(1, len(elems)):
			new_dict[identifiers[j]] = float(elems[j])

		attributes.append(new_dict)

	return attributes


def load_LFW_images(filepath):
	all_files = []

	attr = load_LFW_attributes("../lfw_attributes.txt")
	label_files = {}
	for index, person in enumerate(attr):
		name_list = person["person"].strip().split(" ")
		pic_num = int(person["imagenum"])
		name_list.append("{:04d}".format(pic_num))
		file_name = "_".join(name_list) + ".jpg"
		label_files[file_name] = index

	for path, subdirs, files in os.walk(filepath):
		for name in files:
			if name not in label_files:
				continue
			all_files.append((os.path.join(path, name), label_files[name]))
	all_files = sorted(all_files, key=cmp_to_key(str_cmp))
	all_files = [f[0] for f in all_files]
	all_files = [cv2.imread(f) for f in all_files]
	return all_files

"""
res = load_LFW_images('../lfw/')

cv2.imshow('image', res[1])

cv2.waitKey(0)

cv2.destroyAllWindows()"""



