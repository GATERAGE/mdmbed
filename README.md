# md-embed (c) 2024 web3dguy 

A Python script for processing Markdown files, generating embeddings, and storing them in a vector store. This tool allows you to clean, split, and embed Markdown documents using various methods and embedding models.
Features

    Data Cleaning: Removes duplicates and filters out unwanted content like '404' pages and lines containing the '©' symbol.
    Flexible Input: Supports input from JSON files containing URLs and Markdown data, folders of Markdown files, or single Markdown files.
    Document Splitting: Splits documents using Markdown headers or recursive character splitting.
    Embedding Options: Supports embedding using HuggingFace or Ollama embeddings.
    Vector Store Integration: Stores embeddings in a Chroma vector store for efficient retrieval and analysis.
    Customizable Filters: Option to disable filters that remove specific content.
    Logging: Generates logs for duplicates and removed files for better traceability.



Installation
Prerequisites
```bash
    Python 3.7 or higher
    pip
    Git (optional, for cloning the repository)
```
Clone the Repository

```bash

git clone https://github.com/GATERAGE/mdmbed.git
cd mdmbed
```
Install Required Packages

Install the required Python packages using pip:

```bash

pip install -r requirements.txt
```
Note: The requirements.txt file should list all the dependencies, such as tqdm, langchain, chromadb, huggingface, etc.
Usage

Run the script using Python:

```bash

python md-embed.py [--filters-off]
```
# Command-Line Arguments

    --filters-off: Disable filters that remove lines containing '©' and skip files containing both '404' and 'page not found'.

Upon running the script, you will be prompted to choose an input method:

    JSON Input File Containing URLs and Markdown Data
    Folder of Markdown Files
    Single Markdown File

JSON Input File

If you choose Option 1, you will be asked to provide:

    Path of the JSON input file: The file should be a JSON array of objects, each containing url and markdown keys.
    Path of the output folder: The folder where cleaned Markdown files and logs will be saved.

The script will:

    Clean the data by removing duplicates.
    Save the cleaned Markdown files to the specified output folder.
    Generate a file_to_url.json mapping file.
    Display a summary of the processing.

Folder of Markdown Files

If you choose Option 2, you will be asked to provide:

    Path of the folder containing Markdown files.

The script will:

    Load all .md files from the specified folder.
    Optionally filter out unwanted content.
    Proceed to document splitting.

Single Markdown File

If you choose Option 3, you will be asked to provide:

    Path of the Markdown file.

The script will:

    Load the specified Markdown file.
    Optionally filter out unwanted content.
    Proceed to document splitting.

Document Splitting

After loading the documents, you will be prompted to split them:

    Split Method: Choose between markdown or recursive splitting.
    Remove Links: Optionally remove links from the Markdown content.
    Language: Specify the programming language or language of the content.
    Additional Settings:
        For Markdown Splitting:
            Header Levels: Specify which header levels (#, ##, etc.) to split on.
        For Recursive Splitting:
            Chunk Size: Specify the maximum size of each chunk.
            Chunk Overlap: Specify the number of overlapping characters between chunks.

You will have the option to preview the split data before proceeding.
Embedding and Saving

After splitting, you will be prompted to embed and save the documents:

    Embedding Method: Choose between huggingface or ollama.
        HuggingFace: Enter the embedding model name (default: all-MiniLM-L6-v2).
        Ollama: Enter the Ollama model name (default: nomic-embed-text).
    Persist Directory: Specify the directory to save the vector store database.
    Collection Name: Enter a name for the Chroma collection.

The script will:

    Embed the documents using the chosen embedding method.
    Save the embeddings to a Chroma vector store.
    Display information about the saved collections.

Examples
Example 1: Process JSON Input File

```bash

python md-embed.py
```
Choose Input Method: 1

    Enter the path of the JSON input file: ./data/input.json
    Enter the path of the output folder: ./output

Proceed through the prompts to clean data, split documents, and embed them.
Example 2: Process Folder of Markdown Files with Filters Off

```bash

python md-embed.py --filters-off
```
Choose Input Method: 2

    Enter the path of the folder containing markdown files: ./markdown_files

Proceed through the prompts to load, split, and embed the documents.
Contributing

Contributions are welcome! Please follow these steps:

    Fork the repository.

    Create a new branch:

```bash

git checkout -b feature/your-feature-name
```
Make your changes and commit them:

```bash
git commit -m "Add your message"
```
Push to the branch:

```bash
git push origin feature/your-feature-name
```
    Open a Pull Request.

Please make sure your code adheres to the existing style and that all tests pass.
License

This project is licensed under the MIT License.
Acknowledgments
    web3dguy
    LangChain for text splitting and document handling.
    HuggingFace for embedding models.
    Chroma for the vector store.
    TQDM for progress bars.
    The open-source community for continuous support and contributions.

Disclaimer: This tool is provided "as is" without warranty of any kind. Use it at your own risk. Open source or go away.
