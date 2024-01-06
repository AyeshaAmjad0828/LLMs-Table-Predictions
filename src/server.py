import time
import threading
from flask import Flask, jsonify, request, send_file#Response
import argparse
import importlib
import os
import shutil
import tempfile
import zipfile
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
DEPLOYMENT_FOLDER = 'deployments/'
is_training = False
training_completed = False
module = None
generate_model = None
process_id = None
config_path = "/configs"
config_file = "falcon_40b/config-7b-lora-ae.yml"
config_file_for_predict = "config-7b-lora-ae-for-predict.yml"

def train_model_thread(data):
    global module, is_training, training_completed
    
    try:
        script_name = 'finetune'
        module = importlib.import_module(f'{script_name}')
    except Exception as e:
        print(f"Error while trying to load module: {e}")
        return
    
    try:
        training_completed = False
        is_training = True
        module.train(os.path.join(config_path, config_file), False, **data) 
    except Exception as e:
        print(f"Error while finetuning: {e}")
        stack_trace = traceback.format_exc()
        print(stack_trace)
    
    finally:
        is_training = False
        training_completed = True

        
@app.route('/train', methods=['POST'])
def train_model():
    global is_training
    global process_id
    
    if is_training:
        return jsonify({'message': 'Model finetuning already in progress. Please wait for it to finish.'})
    
    data = request.get_json()
    print(f'Data Received: {data}')
    process_id = data['ID']
    training_thread = threading.Thread(target=train_model_thread, args=(data,))
    training_thread.start()

    print("Model finetuning starting...")
    return jsonify({'message': 'Model finetuning starting...'})


@app.route('/monitor', methods=['GET'])
def monitor():
    global is_training, training_completed, process_id
    output_dir = '../models/falcon-7b-ae'
    if training_completed:
        return jsonify(module.monitor(process_id))

    if is_training == False or module is None:
        return jsonify({'message': 'No model is training.'})
    
    return jsonify(module.monitor(process_id, 0.04))
    
@app.route('/predict', methods=['POST'])
def predict():
    global generate_model
    
    data = request.get_json()
    print(f'Data Received: {data}')

    output_dir = data['DestinationDirectory'] if data['DestinationDirectory'] is not None else '../models/falcon-7b-ae'
    
    try:
        script_name = 'inference'
        module = importlib.import_module(f'{script_name}')
    except Exception as e:
        return jsonify({'message': f"Error while trying to load module: {e}"})

    try:
        if generate_model is None:
            generate_model = module.get_model("../models/falcon-7b-ae/merged")
            #del "../models/mpt-alpaca-7b-ae/merged"
        print("Inferring")
        result = module.predict(get_prompt(data['Input']), generate_model)
        print(result)
        return jsonify({'result': result})
    except Exception as e:
        print(e)
        return jsonify({'message': f"Error while predicting: {e}"})

    
def get_prompt(instruction):
    prompt_template = f"Below is an instruction that describes a task. \n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt_template 


@app.route('/system_info', methods=['GET'])
def get_system_info():
    import torch
    
    system_health = dict()
    
    try:
        script_name = 'system_info'
        module = importlib.import_module(f'{script_name}')    
    except Exception as e:
        return jsonify({'message': f"Error while trying to load module: {e}"})
    
    
    try:
        system_health['Disk'] = module.get_disk_details()
        system_health['Memory'] = module.get_memory_details()
        system_health['CPU'] = module.get_cpu_details()
        if torch.cuda.is_available():
            system_health['GPU'] = module.get_gpu_details()
        
        return jsonify(system_health)
    except Exception as e:
        print(e)
        return jsonify({'message': f"Error: {e}"})        


@app.route('/', methods=['GET'])
def health_check():
    import torch
    import bitsandbytes as bnb

    result = {
        'gpu_available': torch.cuda.is_available(),
        'status': 'healthy',
    }
    return jsonify(result)

        
@app.route('/test', methods=['GET'])
def test():
    data = {
        "LLMType": 0,
        "Name": "tiiuae/falcon-7b",
        "LLMPrompter": "AlpacaPrompter",
        "base_model": "tiiuae/falcon-7b",
        "adapter": "lora",
        "path": "uploads/source.jsonl",
        "ID": 1000,
    }
    global is_training
    global process_id
    
    if is_training:
        return jsonify({'message': 'Model finetuning already in progress. Please wait for it to finish.'})
    
    process_id = data['ID']
    training_thread = threading.Thread(target=train_model_thread, args=(data,))
    training_thread.start()

    print("Model finetuning starting through test endpoint")
    return jsonify({'message': 'Model finetuning starting...'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9262)
    
#2xkzt3uu32gcmeaosjtitw6emvlp6uc4xrxdkne52qtc3hdmddcq



