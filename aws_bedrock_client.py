# aws_bedrock_client.py
import boto3
from typing import List
import json
import re

import boto3.session

# AWS Bedrock setup 
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')  # Adjust region if needed

class QuestionPair:
    def __init__(self, question: str, topic: str):
        self.question = question
        self.topic = topic

class EvaluationPair:
    def __init__(self, question: str, topic: str, answer: str):
        self.question = question
        self.topic = topic
        self.answer = answer

def call_claude_model(prompt: str, model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0") -> str:
    response = bedrock.invoke_model(
        modelId=model_name,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",  # Use 'user' role to provide input
                    "content": f"You are an elementary school teacher. {prompt}"
                }
            ]
        })
    )

    # Read and decode the StreamingBody
    response_body = response["body"].read().decode('utf-8')
    print("Raw response:", response)

    result = json.loads(response_body)
    print(result)
    return result

# Function to generate questions using the transcript
def generate_questions_from_transcript(transcript: str) -> List[QuestionPair]:
    prompt = f"""Generate as many relevant questions as possible on the underlying topic based on the provided video transcript. Each question should be prefixed with the topic name in square brackets, and the questions should be listed without any introductory phrases. Separate each question with a newline character. Use the following schema:

[TOPIC] 1. Question 1
[TOPIC] 2. Question 2
[TOPIC] 3. Question 3
...
[TOPIC] N. Question N

Here is the video transcript:
{transcript}
"""
    generated_response = call_claude_model(prompt)

    # Extracting the generated questions from the content field
    content = generated_response.get('content', [])
    
    question_pairs = []
    
    if isinstance(content, list):
        for item in content:
            if 'type' in item and item['type'] == 'text':
                questions = item['text'].strip().split('\n')
                for question in questions:
                    question = question.strip()
                    if question:
                        # Extract the topic using regex
                        topic_match = re.match(r'\[(.*?)\]', question)  # Match the text within brackets
                        if topic_match:
                            topic = topic_match.group(1)  # Extract the topic
                            question_text = question.replace(topic_match.group(0), '').strip()  # Remove the topic from the question
                        else:
                            topic = "Unknown"  # Default topic if not found
                            question_text = question

                        question_pairs.append(QuestionPair(question=question_text, topic=topic))

    return question_pairs


def evaluate_responses(pairs: List[EvaluationPair], system_prompt: str) -> str:
    pairs_text = "\n".join([f"Question: {pair.question}\nAnswer: {pair.answer}\nTopic: {pair.topic}" for pair in pairs])
    full_prompt = f"{system_prompt}\n\n{pairs_text}\n\nPlease evaluate the student responses."

    response = call_claude_model(full_prompt)
    print(response)
    return response.get('content', {})
