import openai
import pandas as pd
from tqdm import tqdm

def GPT_get_helpfulness(client, question, gold, output):
    score = 0
    prompt = [{'role': 'user', 'content': f'Questions:\n{question}\nResponse A:\n{gold}\nResponse B:\n{output}\nWhich response is more helpful, A or B. ANSWER WITH ONLY ONE LETTER: '}]

    for prompt in prompts:
        try:
            r = client.chat.completions.create(model='gpt-4o', messages=[prompt])\
                      .choices[0].message.content

            r = r.strip()
            a = r.rfind('A')
            b = r.rfind('B')

            if b > a:
                score = 1
            else:
                score = 0
                
        except Exception as a:
            print('Error: ', a)
            score = 0.5

    return score


def win_rate(alpaca_path, model_outputs):
    client = openai.OpenAI(api_key = '')    
    alpaca = pd.read_json(alpaca_path)
    num_tests = len(alpaca)
    score = 0

    for i in tqdm(range(num_tests)):
        score += GPT_get_helpfulness(client, alpaca.iloc[i]['instruction'], alpaca.iloc[i]['output'], model_outputs[i])

    return score / num_tests * 100


    