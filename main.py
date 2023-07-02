from alpha_vantage.timeseries import TimeSeries
from flask import Flask, make_response

from models.linear_regression.linear_regression_main import linear_regression_prediction
from models.support_vector_machine.support_vector_machine_main import support_vector_machine_prediction
from models.decision_tree.decision_tree_main import decision_tree_regression_prediction
from models.neural_network.neural_network_main import neural_network_prediction
from shared.secrets import API_KEY

app = Flask(__name__)


@app.route('/linear/<company_code>/<period>', methods=['GET'])
def linear_regression(company_code, period):
    body = linear_regression_prediction(company_code, period)
    response = make_response(body, 200)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

  
@app.route('/decision-tree/<company_code>/<period>', methods=['GET'])
def decision_tree(company_code, period):
    body = decision_tree_regression_prediction(company_code, period)
    response = make_response(body, 200)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

  
@app.route('/neural-network/<company_code>/<period>', methods=['GET'])
def neural_network(company_code, period):
    body = neural_network_prediction(company_code, period)
    response = make_response(body, 200)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/svm/<company_code>/<period>', methods=['GET'])
def support_vector_machine(company_code, period):
    body = support_vector_machine_prediction(company_code, period)
    response = make_response(body, 200)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    app.run()
