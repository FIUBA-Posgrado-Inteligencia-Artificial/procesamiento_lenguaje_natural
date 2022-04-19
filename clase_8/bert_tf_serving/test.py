import requests
import json
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

max_length = 140
input_text = "I love this app"

tf_batch = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=max_length,
    return_token_type_ids=False,
    return_attention_mask=True,
    padding="max_length",
    truncation=True,
)

#TF Serve endpoint
url = "http://localhost:8501/v1/models/mybert:predict"

payload = {"instances": [{"input_ids": tf_batch['input_ids'], "attention_mask": tf_batch['attention_mask']}]}

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
print(json.loads(response.text)['predictions'])