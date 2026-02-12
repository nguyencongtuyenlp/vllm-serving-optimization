"""
Example script showing how to make direct API calls to vLLM server.
"""

import requests
import json

def test_models_endpoint():
    """Test /v1/models endpoint."""
    response = requests.get('http://localhost:8000/v1/models')
    print("Available models:")
    print(json.dumps(response.json(), indent=2))

def test_completion():
    """Test /v1/completions endpoint."""
    payload = {
        "prompt": "Explain what is machine learning in one sentence:",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = requests.post(
        'http://localhost:8000/v1/completions',
        json=payload
    )
    
    print("\nCompletion result:")
    print(json.dumps(response.json(), indent=2))

def test_streaming():
    """Test streaming completion."""
    payload = {
        "prompt": "Write a haiku about programming:",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": True
    }
    
    print("\nStreaming completion:")
    with requests.post(
        'http://localhost:8000/v1/completions',
        json=payload,
        stream=True
    ) as response:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() != '[DONE]':
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            text = data['choices'][0].get('text', '')
                            print(text, end='', flush=True)
    print("\n")

if __name__ == '__main__':
    print("Testing vLLM OpenAI API\n" + "="*50)
    
    test_models_endpoint()
    test_completion()
    test_streaming()
    
    print("\n" + "="*50)
    print("All tests completed!")
