import numpy as np
import pandas as pd

predictions = np.loadtxt('../fair_face_predictions.csv', delimiter=',')
fair_att = pd.read_csv('../fairface_label_val.csv')
fair_att['attractive'] = predictions
race_data = fair_att['race'].value_counts().reset_index()
attrac = fair_att[fair_att['attractive'] == 1].reset_index(drop=True)
attr_rac_data = pd.DataFrame(attrac['race'].value_counts()).reset_index()
race_data['attractive_race_data'] = attr_rac_data['race']
race_data = race_data.rename(columns={'index': 'race', 'race': 'race_data'})
race_data['attractive_percentages'] = race_data['attractive_race_data']/race_data['race_data']
print(race_data)