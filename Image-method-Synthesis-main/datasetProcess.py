import json
import numpy as np
from transformation import matchJD2toJD1
import os

# Open the JSON file in read mode
with open('BSIdict_468_062324_3.json', 'r') as file:
    # Load the JSON data into a Python dictionary
    BSIdict_468 = json.load(file) 

with open('KV_468_062324.json', 'r') as file2:
    # Load the JSON data into a Python dictionary
    mechStackKV = json.load(file2)

with open('VK_468_062324.json', 'r') as file3:
    # Load the JSON data into a Python dictionary
    mechStackVK = json.load(file3)


four_bar = ['RRRR', 'RRRP', 'RRPR', 'PRPR'] 
six_bar  = ['Watt1T1A1', 'Watt1T2A1', 'Watt1T3A1', 'Watt1T1A2', 'Watt1T2A2', 'Watt1T3A2', 
            'Watt2T1A1', 'Watt2T2A1', 'Watt2T1A2', 'Watt2T2A2', 'Steph1T1', 'Steph1T2',
            'Steph1T3', 'Steph2T1A1', 'Steph2T2A1', 'Steph3T1A1', 'Steph3T2A1', 'Steph3T1A2', 
            'Steph3T2A2', 'Steph2T1A2', 'Steph2T2A2']
eight_bar = list(set(mechStackKV.keys()) - set(four_bar) - set(six_bar))

watt1 = ['Watt1T1A1', 'Watt1T2A1', 'Watt1T3A1', 'Watt1T1A2', 'Watt1T2A2', 'Watt1T3A2']
watt2 = ['Watt2T1A1', 'Watt2T2A1', 'Watt2T1A2', 'Watt2T2A2']
steph1= ['Steph1T1', 'Steph1T2', 'Steph1T3'] 
steph2= ['Steph2T1A1', 'Steph2T2A1', 'Steph2T1A2', 'Steph2T2A2']
steph3= ['Steph3T1A1', 'Steph3T2A1', 'Steph3T1A2', 'Steph3T2A2']

prefix = '062324-' 
directory = './outputs-'

# These fixes should not appear at all. 
# Remove this after Wei's fix. (or not? The last number in S determines which link the slot is)
BSIdict_468['RRRP']['S'] = [[2, 5, 3, 3]] 
BSIdict_468['RRPR']['S'] = [[2, 5, 3, 3], [2, 6, 3, 3]] 
BSIdict_468['PRPR']['S'] = [[0, 1, 2, 0], [1, 3, 4, 1]] 


def getFileString(mechString, filetype = 'z', prefix = prefix, kv = mechStackKV, directory = directory):
    indexString = kv[mechString]
    return directory + filetype + '/' + prefix + filetype + '-' + str(indexString) + '.npy'


def stackMechs(selections):
    # Stack all necessary data
    selections = set(selections)
    if any('all' in tup for tup in selections):  # if that element is in selections 
        selections = set(list(selections) + four_bar + six_bar + eight_bar)
        selections.remove('all')
    if any('four_bar' in tup for tup in selections):  # if that element is in selections 
        selections = set(list(selections) + four_bar)
        selections.remove('four_bar')
    if any('six_bar' in tup for tup in selections):  # if that element is in selections 
        selections = set(list(selections) + six_bar)
        selections.remove('six_bar')
    if any('eight_bar' in tup for tup in selections):  # if that element is in selections 
        selections = set(list(selections) + eight_bar)
        selections.remove('eight_bar')
    if any('watt1' in tup for tup in selections): 
        selections = set(list(selections) + watt1)
        selections.remove('watt1')
    if any('watt2' in tup for tup in selections): 
        selections = set(list(selections) + watt2)
        selections.remove('watt2')
    if any('steph1' in tup for tup in selections): 
        selections = set(list(selections) + steph1)
        selections.remove('steph1')
    if any('steph2' in tup for tup in selections): 
        selections = set(list(selections) + steph2)
        selections.remove('steph2')
    if any('steph3' in tup for tup in selections): 
        selections = set(list(selections) + steph3)
        selections.remove('steph3')
    fileStringsZ = []
    #print(selections)
    for mechString in selections: 
        #print(mechString)
        fileStringsZ.append(getFileString(mechString))
    
    bigZ = []
    list_indices = []
    original_indices = []

    for fileStringZ in fileStringsZ: 
        try:
            if os.path.exists(fileStringZ):
                data = np.load(fileStringZ).tolist()
                bigZ = bigZ + data
                list_indices = list_indices + [int(fileStringZ.split('-z-')[-1].split('.npy')[0])] * len(data)
                original_indices = original_indices + list(range(len(data)))
            else:
                print(f"File '{fileStringZ}' does not exist, skipping.")
        except Exception as e:
            print(f"An error occurred: {e}")

    return bigZ, list_indices, original_indices


localsave = True #if you want to test 
# Get the corresponding mechanism after query 
def getMech(bigZ_index, list_indices, original_indices, param1, BSIdict = BSIdict_468, vk = mechStackVK, saveToLocal = localsave): 
    mechString = vk[str(int(list_indices[bigZ_index]))]
    pos  = original_indices[bigZ_index]
    fileString = getFileString(mechString, filetype = 'encoded')
    pack = np.load(fileString)[pos, :]
    param2 = np.array(pack[-9:])
    mechKey = vk[str(int(pack[-10]))] # -10 because the transformation matrix size is 3x3 
    jd2 = np.array(pack[:-10]).reshape((-1, 2))
    pN = matchJD2toJD1(jd2, param1, param2)
    BSIpc = {
            "B": BSIdict[mechKey]['B'],
            "S": BSIdict[mechKey]["S"],
            "I": BSIdict[mechKey]["I"],
            "p": pN.tolist(),
            "c": BSIdict[mechKey]["c"],

    }
    
    """
    #print(len(BSIpc['p']), len(BSIpc['B'][0]), mechKey)
    if saveToLocal:
        file_path = './sample.json'
        # Open the file in write mode and write the data
        with open(file_path, 'w') as file:
            json.dump(BSIpc, file)  # 'indent=4' makes the output pretty-printed
        localsave = False
    """
    return BSIpc