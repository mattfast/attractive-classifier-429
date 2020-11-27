

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