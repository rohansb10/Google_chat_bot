# Google_chat_bot

[Watch the demo video](https://github.com/rohansb10/Google_chat_bot/blob/main/show_video.mp4)

![project demo web image](https://github.com/rohansb10/Google_chat_bot/blob/main/demo_img.png) 

This repository contains the code for an advanced chatbot built using LangChain, HuggingFace's Mistral-Nemo-Instruct-2407 model, and Streamlit. The chatbot is capable of searching Wikipedia, Arxiv, and DuckDuckGo, performing mathematical calculations, and responding to user queries in real-time. The responses are formatted for easy readability, and the interactions are logged to a CSV file.

## Features

- **Web Search**: Perform real-time web searches using DuckDuckGo.
- **Wikipedia Search**: Fetch and display Wikipedia content.
- **Arxiv Search**: Retrieve research papers and summaries from Arxiv.
- **Calculator**: Perform basic mathematical calculations.
- **Advanced LLM Integration**: Utilize HuggingFace's Mistral-Nemo-Instruct-2407 model for complex queries.
- **Conversation Memory**: Keep track of the conversation context.
- **Response Formatting**: Enhanced readability with bullet-point formatting.
- **CSV Logging**: Logs all interactions to a CSV file, including input, output, and response time.

## Installation

### Prerequisites

- Python 3.8+
- A Hugging Face account and API token
- A LangChain API key
- Streamlit

### Clone the Repository

```bash
git clone https:[//github.com/yourusername/advanced-chatbot](https://github.com/rohansb10/Google_chat_bot)
cd advanced-chatbot
```

### Install the Required Packages

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```plaintext
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langchain_api_key
```

## Run the App

To run the Streamlit app locally, use:

```bash
streamlit run langgraph_chatbot_with_tools.py
```

## Usage

1. **User Input**: Enter a query or a calculation in the input box at the bottom of the app.
2. **Response**: The chatbot will process the input, perform the necessary actions, and return a formatted response.
3. **Tool Selection**: Depending on the query, different tools (web search, Wikipedia, Arxiv, calculator) will be used to fetch or calculate the required information.
4. **Logging**: All interactions are saved in `agent_interaction_log.csv`.

## Project Structure

```plaintext
├── app.py                           # Main application script
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example environment file
├── agent_interaction_log.csv        # Log file for interactions (auto-generated)
└── README.md                        # Project README file
```

## Example Queries

- "What is the capital of France?"
- "Search Wikipedia for Quantum Computing."
- "Find the latest research on AI from Arxiv."
- "What is 3+5?"

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes or improvements.

