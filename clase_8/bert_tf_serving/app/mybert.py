import json
import requests
import numpy as np

#import tensorflow as tf
from transformers import BertTokenizer

class MyBertModel():
    def __init__(self):
        self.max_length = 140
        self.class_names = ['negative', 'neutral', 'positive']
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def predict(self, input_text):

        # in this case, NOT use tensor vector
        # as data will be send by HTTP as JSON string type
        batch = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length, # truncates if len(s) > max_length
            return_token_type_ids=False,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
        )

        #TF Serve endpoint
        url = "http://127.0.0.1:8501/v1/models/mybert:predict"

        payload = {"instances": [{"input_ids": batch['input_ids'], "attention_mask": batch['attention_mask']}]}

        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        y_prob_ensayo = json.loads(response.text)['predictions']
        y_prob = np.argmax(y_prob_ensayo, axis=1)
        class_predicted = self.class_names[int(y_prob)]
        print("Input:", input_text)
        print("Clasificacion:", class_predicted)
        return class_predicted