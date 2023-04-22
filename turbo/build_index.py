import numpy as np
import openai
import os
import json
import textwrap
import re


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def gpt3_embedding(content, engine='text-embedding-ada-002'):
	response = openai.Embedding.create(input=content, engine=engine)
	vector = response['data'][0]['embedding']
	return vector
	
openai.api_key = open_file('openai_apikey.txt')

if __name__ == '__main__':
    alltext = open_file('input.txt')
    chunks = textwrap.wrap(alltext, 3000)
    result = list()
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding = 'ASCII', errors = 'ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        print(info, '\n\n\n')
        result.append(info)
    with open('index.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(result, outfile, indent = 2, ensure_ascii=False)