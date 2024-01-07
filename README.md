# Using LLMs to Predict on Tabular Data
## Finetuning LLMs to predict on some of the popular Kaggle datasets. 

> Refer to the [pre-requisites]([AyeshaAmjad0828/LLMs-Table-Predictions: Training LLMs on some of the popular kaggle datasets (github.com)](https://github.com/AyeshaAmjad0828/LLMs-Table-Predictions?tab=readme-ov-file#pre-requisites)) section to set up the environment for this experiment.

### Technology Stack

| Hugging Face                          | Axolotl                                                      | Modal                                                        | Llama                                                        | OpenAI                                                       | Weights&Biases                                          |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| ![images](./README.assets/images.png) | ![image-20240107212039440](./README.assets/image-20240107212039440.png) | ![image-20240107211831323](./README.assets/image-20240107211831323.png) | ![image-20240107212749542](./README.assets/image-20240107212749542.png) | ![image-20240107212933043](./README.assets/image-20240107212933043.png) | ![wb-logo-lightbg](./README.assets/wb-logo-lightbg.png) |

### Overview

The objective of this project is to assess the quality of LLM training and predictions results on tabular datasets. The inspiration is drawn from [clinicalml/TabLLM (github.com)](https://github.com/clinicalml/TabLLM) which performed few-shot classification of tabular data with LLMs. 

The project consists of two experiments:

1. Finetuning an LLM on serialized training data and using it to generate predictions on test dataset. 
2. Generating vector embeddings on serialized training data and using the vectors as features for an AutoML algorithm(FLAML). 

Let's go over the design and execution of each experiment in a step by step manner. 

### Experiment 1 - Finetuning an LLM

Here is an a diagram showing a high-level set up of the first experiment.

![Experiment1](./README.assets/Experiment1.jpg)

This experiment contains three steps:

1. Serializing the data (both train and test) from table to text. (list serialization, text template, manual template)
2. Finetuning an LLMs (Llama-7B, Llama-13b, Llama 70b). All logic related to finetuning is in src/finetune.py
3. Perform inference/prediction on test data using finetuned model. All logic related to inference is in src/inference.py

#### Table to Text



### Experiment 2 - LLM Vector Embeddings for AutoML

Here is an a diagram showing a high-level set up of the second experiment.

![Experiment2](./README.assets/Experiment2.jpg)



### Pre-requisites

1. Create an account on [Modal](https://modal.com/).

2. Create an account on [Hugging face](https://huggingface.co/) and agree to the terms and conditions for accessing [Llama](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) models. 

3. Get the [hugging face access token](https://huggingface.co/settings/tokens). 

4. Create a new [secret](https://modal.com/ayeshaamjad0828/secrets) for hugging face in your modal account. This secret is a way to mask [hugging face access token](https://modal.com/ayeshaamjad0828/secrets). 

   ![image-20240107203301672](./README.assets/image-20240107203301672.png)

   Once created, your keys will be displayed in the same location. 
   ![image-20240107203439593](./README.assets/image-20240107203439593.png)

5. Install modal in your current python environment `pip install modal`.

6. Open cmd, navigate to python scripts folder  ...\AppData\Local\Programs\Python\Python310\Scripts

7.  Set up modal token in your python environment `modal setup`.

   ![modal-setup](./README.assets/modal-setup.PNG)

8. (Optional) To monitor LLM finetuning performance visually, set up a [weights and biases account](https://wandb.ai/home) , get its [authorize key](https://wandb.ai/authorize), and create its [secret](https://modal.com/ayeshaamjad0828/secrets) in the same way as hugging face secret on modal. 

   Install weights and biases library in your current python environment  `pip install wandb`

   Add your wandb config to your config.yml script (you will find this in my exampleconfig.yaml)

   ```python
   wandb_project: code-7b-sql-output
   wandb_watch: all
   wandb_entity:
   wandb_run_id:
   ```

> you may have to perform modal setup again in your python environment as shown in step 7. 

9. Add both hugging face and weights and biases secrets to common.py script for initializing the stub:

   ```python
   stub = Stub(APP_NAME, secrets=[Secret.from_name("my-huggingface-secret1"), Secret.from_name("my-wandb-secret1")])
   ```

   
