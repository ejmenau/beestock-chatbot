# BeeStock WMS RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed to provide intelligent support for BeeStock WMS (Warehouse Management System) based on customer documentation and processes.

## 🚀 Features

- **Intelligent Document Processing**: Automatically processes and indexes `.docx` files
- **Vector Search**: Uses FAISS for fast similarity search across document chunks
- **AI-Powered Answers**: Leverages Google Gemini Pro for context-aware responses
- **Customer Filtering**: Filter responses by specific customer implementations
- **Source Attribution**: Always shows which documents were used for answers

## 📁 Project Structure

```
beestock-chatbot/
├── build_index.py          # Indexing pipeline - processes documents and builds vector index
├── ask_chatbot.py          # RAG chatbot interface - user-facing application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── chunks_data.json       # Generated: Document chunks with metadata
├── vector_index.faiss     # Generated: FAISS vector index
└── *.docx                 # Source documents (customer processes)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ejmenau/beestock-chatbot.git
   cd beestock-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get your Gemini API key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Update `ask_chatbot.py` with your key

## 🚀 Usage

### Step 1: Build the Index

First, process your documents and build the vector index:

```bash
python build_index.py
```

This will:
- Load all `.docx` files from the directory
- Extract text and metadata
- Create document chunks
- Generate embeddings using SentenceTransformer
- Build and save a FAISS vector index

### Step 2: Start the Chatbot

Once the index is built, start the interactive chatbot:

```bash
python ask_chatbot.py
```

## 💬 Example Questions

You can ask questions like:
- "How do I configure inventory tracking?"
- "What are the warehouse setup requirements?"
- "How do I manage user permissions?"
- "What are the best practices for order processing?"

## 🔧 Configuration

### API Key Setup

In `ask_chatbot.py`, replace the placeholder:
```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

### Customizing the Model

The system uses `BAAI/bge-small-en-v1.5` for embeddings. You can modify this in both scripts if needed.

## 📊 How It Works

1. **Indexing Pipeline** (`build_index.py`):
   - Reads `.docx` files and extracts text
   - Splits documents into manageable chunks
   - Generates vector embeddings using SentenceTransformer
   - Creates a FAISS index for fast similarity search

2. **Retrieval & Generation** (`ask_chatbot.py`):
   - Converts user questions to embeddings
   - Searches the index for similar document chunks
   - Retrieves relevant context
   - Uses Gemini Pro to generate answers based on context

## 🎯 Use Cases

- **Customer Support**: Quick answers to common questions
- **Training**: Onboarding new team members
- **Process Documentation**: Centralized knowledge base
- **Troubleshooting**: Fast access to relevant information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Verify your Gemini API key is correct
3. Ensure you've run `build_index.py` before `ask_chatbot.py`
4. Check the error messages for specific guidance

## 🔮 Future Enhancements

- Web interface
- Multi-language support
- Document upload functionality
- Conversation history
- Export capabilities
- Integration with other systems

---

**Built with ❤️ for BeeStock WMS users**
