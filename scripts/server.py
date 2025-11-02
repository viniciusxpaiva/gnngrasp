# Import flask and datetime module for showing date and time
from flask import Flask, jsonify
from utils import *
import datetime

x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)


# Route for seeing a data
@app.route('/data')
def get_time():

	elems = [
        { 'name': 'A-ASP-26', 'sets': ['GRaSP', 'GASS'] },
        { 'name': 'A-CYS-36', 'sets': ['GRaSP'] },
        { 'name': 'A-GLU-40', 'sets': ['GASS'] },
        { 'name': 'A-THR-42', 'sets': ['GRaSP', 'PUResNet'] },
    ]
	# Returning an api for showing in reactjs
	#return jsonify(elems)

	return {
        'Name':"geek",
        "Age":"25",
        "Date":x,
        "programming":"python"
       }

	
# Running app
if __name__ == '__main__':
	app.run(debug=True)
