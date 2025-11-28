import numpy as np
from datasetProcess import getMech, stackMechs
from normalize import process_mech_051524
from sklearn.neighbors import KDTree
import json 
import torch
from model import VAE
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


def decode(mechErrors, bigZ_indices, list_indices, original_indices, param1):
    solutions = []
    ctr = 0
    for count, bigZ_index in enumerate(bigZ_indices):
        BSIpc = getMech(bigZ_index, list_indices, original_indices, param1)
        BSIpc["error"] = mechErrors[count]
        solutions.append(BSIpc)
        ctr += 1
    result = {"version": "1.1.0", "solutions": solutions} # 1.0.0 was using the entire set. 1.1 is using partially 
    return result


# Precalculate trees. 
"""
def selectTree(mechType = 'four_bar'): 
    # This requires changes if needed    
    if mechType == 'four_bar' or mechType is None: 
        return trees[0], indexPack_four_bar
    if mechType == 'six_bar':
        return indexPack_six_bar[1], indexPack_six_bar
    if mechType == 'eight_bar':
        return indexPack_eight_bar[2], indexPack_eight_bar
    if mechType == 'all':
        return indexPack_all[3], indexPack_all 
"""

#indexPack_four_bar = stackMechs(['four_bar'])   # bigZ, list_indices, original_indices
#indexPack_six_bar = stackMechs(['six_bar'])     # bigZ, list_indices, original_indices
#indexPack_eight_bar = stackMechs(['eight_bar']) # bigZ, list_indices, original_indices
#indexPack_all = [indexPack_four_bar[i] + indexPack_six_bar[i] + indexPack_eight_bar[i] for i in range(3)]

indexPack = stackMechs(['RRRR']) # select dataset
kdt = KDTree(np.array(indexPack[0]))
print(np.array(indexPack[0]).shape)

# Initialization of Neural Network 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latentDim = 25
checkpoint_path = "./anar_lat_25_052324.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model = VAE(latentDim)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

app = Flask(__name__) 
CORS(app)

@app.route("/")
def index():
    return "Flask App now!"


@app.route("/image-based-path-synthesis", methods=["POST"])
def query():
    #print('query started')
    data = json.loads(request.data.decode("utf8").replace("'", '"'))

    knn = data["knn"] # number of solutions (k-nearest-neighbours)
    path = data["path"] # input path point 

    # if empty list then use four-bar tree 
    # mechType = data.get('types') # types of mechanisms desired not implemented as Wei did not make UIs for this function 

    # Get normalized path image and parameters
    matImg, param1, success = process_mech_051524(np.array([path]).swapaxes(0, 1), 0)
    #print(param1)
    if not success:
        return jsonify([])
    
    # Get z of the path image 
    # 05/31/24: you should also get the mse for reconstruction in the returned object 
    images = (
        torch.from_numpy(np.array([[np.array(matImg)[:, :, 0]]])).float().to(device)
    )
    x = model.encoder(images)
    mean, logvar = x[:, : model.latent_dim], x[:, model.latent_dim :]
    z = model.reparameterize(mean, logvar)
    z = z.cpu().detach().numpy()

    # Search and get mechanism indicies
    # kdt, package = selectTree(mechType) # types of mechanisms desired not implemented as Wei did not make UIs for this function 
    bigZdata, list_indices, original_indices = indexPack
    dist, ind = kdt.query(z.reshape((1, -1)), k=knn)
    mechErrors = dist[0]
    bigZ_indices = ind[0]
    result = decode(mechErrors, bigZ_indices, list_indices, original_indices, param1)

    # Saving synthesis result # comment out if you deploy server 
    # json_string = json.dumps(result, indent=4)
    # with open('result_FB.json', 'w') as json_file:
    #    json_file.write(json_string)
    return jsonify(result)


if __name__ == "__main__":
    # debug: Refresh the server whenever code changes are made. 
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080))) 