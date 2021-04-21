from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
xgb = pickle.load(open('xgboost_regression.pkl', 'rb'))

def prepare_data(area, n_quartos, zona):
	# Tratamento
	colunas = ['area', 'quartos', 'zona_leste', 'zona_norte', 'zona_oeste', 'zona_sul']
	is_norte = 1 if zona == 'norte' else 0
	is_sul = 1 if zona == 'sul' else 0
	is_leste = 1 if zona == 'leste' else 0
	is_oeste =  1 if zona == 'oeste' else 0

	dados_entrada = [[np.log1p(area)],	
	                 [n_quartos],	
	                 [is_leste],	
	                 [is_norte],	
	                 [is_oeste],	
	                 [is_sul]]
	dados_entrada=dict(zip(colunas, dados_entrada))
	X=pd.DataFrame(dados_entrada)
	return X


@app.route('/')
def home():
    return render_template('deploy.html')

@app.route('/predict', methods=['POST'])
def predict():
	# Entradas
	#area = 50
	#n_quartos = 3
	#zona = 'sul'
	features = list(request.form.values())
	zona, n_quartos, area = features[0], int(features[1]), int(features[2])
	X = prepare_data(area, n_quartos, zona.lower())
	pred=xgb.predict(X)
	aluguel = np.expm1(pred[0])
	return render_template('deploy.html', prediction_text=aluguel)

if __name__ == "__main__":
	app.run()

