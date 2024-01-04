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
config_path = "../configs"
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
    prompt_template = f"Below is an instruction that describes a task. Write an astera expression in response to complete the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
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

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files['file']
#     filename = file.filename
#     chunk_index = int(request.form['chunk_index'])
#     total_chunks = int(request.form['total_chunks'])

#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
    
#     chunk_filename = f"chunk_{chunk_index}"
#     file.save(os.path.join(UPLOAD_FOLDER, chunk_filename))

#     if chunk_index == total_chunks - 1:
#         uploaded_file_path = reconstruct_file(total_chunks, filename)
#         return uploaded_file_path

#     return 'Chunk received successfully'

# def reconstruct_file(total_chunks, filename):
#     chunks = []
#     for i in range(total_chunks):
#         chunk_filename = f"chunk_{i}"
#         chunk_path = os.path.join(UPLOAD_FOLDER, chunk_filename)
#         with open(chunk_path, 'rb') as chunk_file:
#             chunks.append(chunk_file.read())
#         os.remove(chunk_path)
    
#     uploaded_file_path = f"{UPLOAD_FOLDER}{filename}"

#     with open(uploaded_file_path, 'wb') as final_file:
#         for chunk in chunks:
#             final_file.write(chunk)

#     print(f"File uploaded successfully: {uploaded_file_path}")
#     return uploaded_file_path


# @app.route('/download/<int:id>', methods=['GET'])
# def download_directory(id):
#     directory_path = f'outputs/{id}'
#     temp_dir = tempfile.mkdtemp()

#     # Create a new subdirectory inside the temporary directory to hold the selected files
#     selected_files_dir = os.path.join(temp_dir, 'selected_files')
#     os.mkdir(selected_files_dir)

#     # Get a list of all files in the 'directory_path'
#     files_in_directory = os.listdir(directory_path)

#     # Loop through each file and copy only the 'bin' and 'json' files
#     for file_name in files_in_directory:
#         file_path = os.path.join(directory_path, file_name)
#         if os.path.isfile(file_path) and file_name.endswith(('.bin', '.json')):
#             shutil.copy(file_path, selected_files_dir)

#     # Archive the selected files directory
#     zip_path = os.path.join(temp_dir, 'selected_files.zip')
#     shutil.make_archive(zip_path[:-4], 'zip', selected_files_dir)

#     return send_file(zip_path, as_attachment=True)


# @app.route('/transferfile', methods=['POST', 'GET'])
# def transferfile():

#     """
#     request body for s3:
#     data_s3 = {
#         'dest_obj': 's3',
#         'source_path': 'C:\\Users\\asteraXYZ\\Folder1\\WeightsDir',
#         'destination_path': 's3://<bucket_name>/<path>',
#         'aws_access_key': '<access-keys>',
#         'aws_secret_key': '<access-secret-keys>'
#         }

#     request body for azure blob:
#     data_azure = {
#         'dest_obj': 'azure_blob',
#         'source_path': 'C:\\Users\\asteraXYZ\\Folder1\\WeightsDir',
#         'destination_path': 'https://<storage_account>.blob.core.windows.net/<container_name>',
#         'storage_account_name': '<storage_account>',
#         'access_key': '<access-keys>'
#         }
#     """

#     data = request.get_json()
    
#     try:
#         script_name = 'file_transfer'
#         module = importlib.import_module(f'{script_name}')
#         #data['dest_obj'] == 's3'
        
#     except Exception as e:
#         return jsonify({'message': f"Error while trying to load module: {e}"})
    
#     if os.path.exists(data['source_path']) or os.path.isdir(data['source_path']):
#         try:
#             if data['dest_obj'] == 's3':
#                 module.transfer_directory_to_s3(data['source_path'], data['destination_path'], data['aws_access_key'],
#                                                 data['aws_secret_key'])
#                 return jsonify({'Files transfer to s3 successfully.'})
#             elif data['dest_obj'] == 'azure_blob':
#                 module.transfer_directory_to_azure_blob(data['source_path'], data['destination_path'], data['storage_account_name'],
#                                                         data['access_key'])
#                 return jsonify({'Files transfer to azure blob storage successfully.'})
#         except Exception as e:
#             print(e)
#             return jsonify({'message': f"Error: {e}"})        
#     else:
#          return jsonify({"Source folder does not exist or is not a directory."})
        
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



