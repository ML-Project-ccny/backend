from flask import Flask, request
from model import SimpleConvNetModel
import torch 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the model
# model = pickle.load(open('./model.pkl','rb'))
pytorch_model = SimpleConvNetModel()

## set params from the saved state of model
pytorch_model.load_state_dict(torch.load('model.pt'), strict=False)

pytorch_model.eval()

@app.route('/',methods=['POST'])
def index():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    arr = torch.FloatTensor(data['data'])

    res = pytorch_model(arr)

    print(res)
    return "HELLO"

if __name__ == '__main__':
    app.run(port=5000,debug=True)