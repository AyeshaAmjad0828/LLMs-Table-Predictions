pip install -U openai, pandas, openpyxl, umap-learn 

import os
import pandas as pd
import openai
import umap
from umap import UMAP
import numpy as np

openai.api_key = os.environ['OPENAI_API_KEY']   ##I would recommend you to add add the key to your env variables and acces it here with its name

##Leaving space here to load text data
file_path = 'sampledata.xlsx'
data = pd.read_excel(file_path)

feature_column = 'features'
target_column = 'target'

def get_embedding(text_to_embed):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-similarity-davinci-001",  ##you can use other models 1. `text-similarity-babbage-001`, `text-similarity-curie-001`, `text-embedding-ada-002`
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return embedding

# Retrieve and store the embeddings
embeddings = data[feature_column].apply(get_embedding)


# #embedding_matrix = np.array(embeddings)

# # Instantiate the UMAP object with desired hyperparameters.
# # The n_components parameter defines the size of the reduced dimensionality.
# umap_model = umap.UMAP(n_components=20, random_state=42, n_neighbors=100, min_dist=0.1)


# # Fit the model and transform the embeddings
# reduced_embeddings = umap_model.fit_transform(data["embeddings"].tolist())



# Convert the list of embeddings into a DataFrame
embeddings_df = pd.DataFrame(embeddings.tolist())


# Concatenate the original DataFrame with the embeddings DataFrame
df_with_embeddings = pd.concat([data, embeddings_df], axis=1)


# Concatenate the original DataFrame with the embeddings DataFrame
df_with_embeddings = pd.concat([data, embeddings_df], axis=1)

# Save the updated dataframe to a new Excel file
df_with_embeddings.to_excel('sampledata_embeddings.xlsx', index=False)

print("Embeddings have been added to the DataFrame and saved to a new Excel file.")
