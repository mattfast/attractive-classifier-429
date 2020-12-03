from load_LFW_attributes import load_LFW_attributes
from matplotlib import pyplot as plt
import scipy.stats
import numpy as np

def main():

	attributes = load_LFW_attributes("../lfw_attributes.txt")
	predictions = np.loadtxt('../lfw_non_binary_trained_continuous_predictions.csv', delimiter=',')
	test_indices = np.loadtxt('../test_values.csv', delimiter=',')
	test_indices = test_indices.astype(np.int64)
	attributes = attributes[test_indices]

	white_scores_men, asian_scores_men, black_scores_men, attractive_scores_men = ([] for i in range(4))
	prediction_scores = []
	for index, img in enumerate(attributes):
		if img["Male"] > 0:
			white_scores_men.append(img["White"])
			asian_scores_men.append(img["Asian"])
			black_scores_men.append(img["Black"])
			attractive_scores_men.append(img["Attractive Man"])
			prediction_scores.append(predictions[index])

	m, b, r_val, p_val, stderr = scipy.stats.linregress(white_scores_men, attractive_scores_men)
	print("White vs Attractive - Men")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	#plt.axline((0, b), slope=m, color="black")
	#plt.scatter(white_scores_men, attractive_scores_men, color='b')
	plt.scatter(white_scores_men, prediction_scores, color='r')
	plt.title("White Score vs Attractive Score - Men")
	plt.xlabel("White Score")
	plt.ylabel("Attractive Score")
	plt.show()

	m, b, r_val, p_val, stderr = scipy.stats.linregress(asian_scores_men, attractive_scores_men)
	print("Asian vs Attractive - Men")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	#plt.axline((0, b), slope=m, color="black")
	#plt.scatter(asian_scores_men, attractive_scores_men)
	plt.scatter(asian_scores_men, prediction_scores, color='r')
	plt.title("Asian Score vs Attractive Score - Men")
	plt.xlabel("Asian Score")
	plt.ylabel("Attractive Score")
	plt.show()

	m, b, r_val, p_val, stderr = scipy.stats.linregress(black_scores_men, attractive_scores_men)
	print("Black vs Attractive - Men")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	#plt.axline((0, b), slope=m, color="black")
	#plt.scatter(black_scores_men, attractive_scores_men)
	plt.scatter(black_scores_men, prediction_scores, color='r')
	plt.title("Black Score vs Attractive Score - Men")
	plt.xlabel("Black Score")
	plt.ylabel("Attractive Score")
	plt.show()

	white_scores_women, asian_scores_women, black_scores_women, attractive_scores_women = ([] for i in range(4))
	for img in attributes:
		if img["Male"] < 0:
			white_scores_women.append(img["White"])
			asian_scores_women.append(img["Asian"])
			black_scores_women.append(img["Black"])
			attractive_scores_women.append(img["Attractive Woman"])

	m, b, r_val, p_val, stderr = scipy.stats.linregress(white_scores_women, attractive_scores_women)
	print("White vs Attractive - Women")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	plt.axline((0, b), slope=m, color="black")
	plt.scatter(white_scores_women, attractive_scores_women)
	plt.title("White Score vs Attractive Score - Women")
	plt.xlabel("White Score")
	plt.ylabel("Attractive Score")
	plt.show()

	m, b, r_val, p_val, stderr = scipy.stats.linregress(asian_scores_women, attractive_scores_women)
	print("Asian vs Attractive - Women")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	plt.axline((0, b), slope=m, color="black")
	plt.scatter(asian_scores_women, attractive_scores_women)
	plt.title("Asian Score vs Attractive Score - Women")
	plt.xlabel("Asian Score")
	plt.ylabel("Attractive Score")
	plt.show()

	m, b, r_val, p_val, stderr = scipy.stats.linregress(black_scores_women, attractive_scores_women)
	print("Black vs Attractive - Women")
	print(f"Slope: {m}, R^2: {r_val**2}, P: {p_val}")

	plt.axline((0, b), slope=m, color="black")
	plt.scatter(black_scores_women, attractive_scores_women)
	plt.title("Black Score vs Attractive Score - Women")
	plt.xlabel("Black Score")
	plt.ylabel("Attractive Score")
	plt.show()

if __name__ == '__main__':
	main()