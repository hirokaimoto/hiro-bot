import openai
import json
import numpy as np
import textwrap
import re
from time import time, sleep
import os
from convert_pdf import save_file


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def similarity(v1, v2):
	return np.dot(v1, v2) #/(norm(v1)*norm(v2))
    

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding = 'ASCII', errors = 'ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']
    return vector


def search_index(text, data, count=4):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        if score >= 0.3:
            scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key = lambda d: d['score'], reverse = True)
    return ordered[0:count]

    
def gpt3_completion(prompt, engine = 'text-davinci-003', temp = 0.3, top_p = 1.0, tokens = 2000, freq_pen = 0.25, pres_pen = 0.0, stop = ['<<END>>']):
    max_retry = 3
    retry = 0
    prompt = prompt.encode(encoding = 'ASCII', errors = 'ignore').decode()
    #prompt = prompt
    while True:
        try:
            response = openai.Completion.create(
                engine = engine,
                prompt = prompt,
                temperature = temp,
                max_tokens = tokens,
                top_p = top_p,
                frequency_penalty = freq_pen,
                presence_penalty = pres_pen,
                stop = stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT: \n\n' + prompt + '\n\n=======\n\nRESPONSE: \n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print("Error communicating with OpenAI", oops)
            sleep(1)

def chat_gpt_completion(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.5)
    text = response['choices'][0]['message']['content']
    filename = f"{time()}_chat.txt"
    if not os.path.exists("gpt3_logs"):
        os.makedirs("gpt3_logs")
    save_file(f"gpt3_logs{filename}", f"{str(messages)} \n\n======\n\n {text}")
    return text

def flatten_convo(conversation):
    convo = ''
    for i in conversation:
        convo += f"{i['role'].upper()}: {i['content']}"
    return convo.strip()
    


if __name__ == '__main__':
    convo_length = 30
    openai.api_key = open_file('openai_apikey.txt')
    conversation = list()
    conversation.append({"role": "user", "content": "I am an helpful assistant, that helps answering questions about the user fully, based on the context provided."})
    with open('index.json', 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    while True:
        query = input("\n\nUSER: ")
        conversation.append({"role": "user", "content": query})
        flat = flatten_convo(conversation)
        results = search_index(query, data)
        print(results)
        # aux = '\n\n'.join(results)
        answers = list()
        #Answer the same question for each chunk
        for result in results:
            prompt = open_file('prompt_answer.txt').replace('<<CONTEXT>>', result['content']).replace('<<QUERY>>', query)
            answer = gpt3_completion(prompt)
            print('\n\n', answer)
            answers.append(answer)
        # aux_2 = gpt3_completion(aux)
        #Create a detailed summary of all the answers of the chunks
        all_answers = '\n\n'.join(answers)
        chunks = textwrap.wrap(all_answers, 5000)
        final = list()
        for chunk in chunks:
            prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk)
            summary = gpt3_completion(prompt)
            final.append(summary)
        prompt = open_file("prompt_anticipate.txt").replace('<<INPUT>>', flat)
        anticipation = gpt3_completion(prompt)
        print(f"\n\nANTICIPATION: {anticipation}")
        prompt = open_file("prompt_salience.txt").replace('<<INPUT>>', flat)
        salience = gpt3_completion(prompt)
        print(f"\n\nSALIENCE: {salience}")
        conversation[0]['content'] = f"I am an helpful assistant, that helps answering questions about the user fully, based on the context provided: {final}. I am provided with several answers from other AIs based on the context: {answers}. I will refine the answers into a final answer to the following question: {query}. I will do my best to fulfill my objectives and answer in detail."

        #Create a detailed summary of all the answers of the chunks
        response = chat_gpt_completion(conversation)
        conversation.append({'role': 'assistant', 'content': response})
        print(f"\n\nASSISTANT: {response}")

