# get sample training data
gsutil cp gs://fras/python_state_changes.txt .
gsutil cp gs://fras/250k_rndm_zinc_drugs_clean.txt .
# get sample model weights
gsutil cp -r gs://fras/python_state_changes .
gsutil cp -r gs://fras/zinc_drugs .