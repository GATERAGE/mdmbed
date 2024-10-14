import json
import re
import os
import logging
from tqdm import tqdm
from uuid import uuid4
import argparse
# Updated import statements based on deprecation warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def clean_url(url):
    try:
        url_tail = url.split('/')[-1]
        if not url_tail:
            url_tail = url.split('/')[-2]
        if "#" in url_tail:
            base, fragment = url_tail.split("#")
            section_title = base.replace('.html', '')
            content = fragment.replace('.', ', ')
        else:
            section_title = url_tail.replace('.html', '')
            content = None
        return section_title, content
    except Exception as e:
        logging.error(f"Error in clean_url with URL '{url}': {e}")
        return None, None

def clean_data(input_data, duplicates_log_file):
    processed_entries = {}
    duplicates_count = 0
    duplicates_log = []
    logging.info("Starting data cleaning process.")

    for index, entry in enumerate(input_data):
        try:
            url = entry.get("url", "")
            markdown_text = entry.get("markdown", "")

            if isinstance(markdown_text, str):
                markdown_text = markdown_text.encode('utf-8', 'replace').decode('utf-8')

            section_title, content = clean_url(url)
            if section_title:
                section_title = section_title.lower()
            else:
                logging.warning(f"Empty section title for URL: {url}")
                continue

            if section_title not in processed_entries:
                processed_entries[section_title] = {
                    "section_title": section_title,
                    "markdown": markdown_text,
                    "url": url  # Include the URL here
                }
            else:
                duplicates_count += 1
                duplicates_log.append((url, index + 1))
        except Exception as e:
            logging.error(f"Error processing entry {index + 1}: {e}")

    if duplicates_log:
        try:
            logs_folder = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_folder, exist_ok=True)
            duplicates_log_file = os.path.join(logs_folder, os.path.basename(duplicates_log_file))
            with open(duplicates_log_file, "w", encoding="utf-8") as log_file:
                log_file.write("Duplicates Removed:\n")
                for duplicate, line_number in duplicates_log:
                    log_file.write(f"Line {line_number}: {duplicate}\n")
        except Exception as e:
            logging.error(f"Error writing duplicates log: {e}")

    logging.info("Data cleaning process completed.")
    return list(processed_entries.values()), duplicates_count

def sanitize_filename(filename):
    try:
        filename = re.sub(r'[\\/*?"<>|\[\]#]', "", filename)
        filename = re.sub(r'[:]', "-", filename)
        filename = re.sub(r'\s+', "_", filename)
        filename = re.sub(r'_{2,}', "_", filename)
        filename = filename.strip('_')
        return filename[:255]
    except Exception as e:
        logging.error(f"Error sanitizing filename '{filename}': {e}")
        return "invalid_filename"

def save_output(output_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    file_to_url = {}

    for entry in tqdm(output_data, desc="Saving Markdown files"):
        try:
            section_title = sanitize_filename(entry["section_title"])
            markdown_text = entry.get("markdown", "")
            url = entry.get("url", "")

            file_name = f"{section_title}.md"
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(markdown_text.strip())

            # Build the mapping from file_name to url
            file_to_url[file_name] = url

        except Exception as e:
            logging.error(f"Error saving file '{file_name}': {e}")

    # Save the file_to_url mapping
    mapping_file_path = os.path.join(output_folder, "file_to_url.json")
    try:
        with open(mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(file_to_url, f)
    except Exception as e:
        logging.error(f"Error saving file to URL mapping: {e}")

def display_summary(changes_made, duplicates_count):
    print(f"Total entries processed: {changes_made}")
    print(f"Total duplicates removed: {duplicates_count}")

def load_markdown_files(output_folder, filters_off=False):
    markdown_files = [f for f in os.listdir(output_folder) if f.endswith('.md')]
    documents = []
    removed_files = []

    # Load the file_to_url mapping if available
    mapping_file_path = os.path.join(output_folder, "file_to_url.json")
    file_to_url = {}
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, "r", encoding="utf-8") as f:
                file_to_url = json.load(f)
        except Exception as e:
            logging.error(f"Error loading file to URL mapping: {e}")

    with tqdm(markdown_files, desc="Loading Markdown files") as pbar:
        for md_file in pbar:
            try:
                file_path = os.path.join(output_folder, md_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not filters_off:
                    # Make content lowercase for case-insensitive matching
                    content_lower = content.lower()
                    # Check for both "404" and "page not found"
                    if "404" in content_lower and "page not found" in content_lower:
                        logging.info(f"Skipping file '{md_file}' due to '404' and 'page not found' in content.")
                        url = file_to_url.get(md_file, "")
                        removed_files.append((md_file, url or "N/A", "Contains both '404' and 'page not found'"))
                        continue

                    # Remove lines containing © symbol
                    content_lines = content.split('\n')
                    content_lines = [line for line in content_lines if '©' not in line]
                    content = '\n'.join(content_lines)

                # Get the URL from the mapping if available
                url = file_to_url.get(md_file, file_path)  # Use file path if URL is not available
                # Create a Document with the content and metadata
                doc = Document(page_content=content, metadata={"source": url})
                documents.append(doc)
                pbar.set_description(f"Processing: {md_file}")
            except Exception as e:
                logging.error(f"Error loading file '{md_file}': {e}")

    # Write removed files to log
    if removed_files:
        try:
            logs_folder = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_folder, exist_ok=True)
            removed_files_log_file = os.path.join(logs_folder, "removed_files.log")
            with open(removed_files_log_file, "w", encoding="utf-8") as log_file:
                log_file.write("Files Removed During Loading Due to Filters:\n")
                for md_file, url, reason in removed_files:
                    log_file.write(f"File: {md_file}, URL: {url}, Reason: {reason}\n")
        except Exception as e:
            logging.error(f"Error writing removed files log: {e}")

    return documents

def load_markdown_files_from_folder(folder_path, filters_off=False):
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    documents = []
    removed_files = []

    with tqdm(markdown_files, desc="Loading Markdown files") as pbar:
        for md_file in pbar:
            try:
                file_path = os.path.join(folder_path, md_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not filters_off:
                    # Make content lowercase for case-insensitive matching
                    content_lower = content.lower()
                    # Check for both "404" and "page not found"
                    if "404" in content_lower and "page not found" in content_lower:
                        logging.info(f"Skipping file '{md_file}' due to '404' and 'page not found' in content.")
                        removed_files.append((md_file, "N/A", "Contains both '404' and 'page not found'"))
                        continue

                    # Remove lines containing © symbol
                    content_lines = content.split('\n')
                    content_lines = [line for line in content_lines if '©' not in line]
                    content = '\n'.join(content_lines)

                # Create a Document with the content and metadata
                doc = Document(page_content=content, metadata={"source": file_path})
                documents.append(doc)
                pbar.set_description(f"Processing: {md_file}")
            except Exception as e:
                logging.error(f"Error loading file '{md_file}': {e}")

    # Write removed files to log
    if removed_files:
        try:
            logs_folder = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_folder, exist_ok=True)
            removed_files_log_file = os.path.join(logs_folder, "removed_files.log")
            with open(removed_files_log_file, "w", encoding="utf-8") as log_file:
                log_file.write("Files Removed During Loading Due to Filters:\n")
                for md_file, url, reason in removed_files:
                    log_file.write(f"File: {md_file}, URL: {url}, Reason: {reason}\n")
        except Exception as e:
            logging.error(f"Error writing removed files log: {e}")

    return documents

def load_single_markdown_file(file_path, filters_off=False):
    documents = []
    removed_files = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not filters_off:
            # Make content lowercase for case-insensitive matching
            content_lower = content.lower()
            # Check for both "404" and "page not found"
            if "404" in content_lower and "page not found" in content_lower:
                logging.info(f"Skipping file '{file_path}' due to '404' and 'page not found' in content.")
                removed_files.append((file_path, "N/A", "Contains both '404' and 'page not found'"))
                return documents  # Empty list

            # Remove lines containing © symbol
            content_lines = content.split('\n')
            content_lines = [line for line in content_lines if '©' not in line]
            content = '\n'.join(content_lines)

        # Create a Document with the content and metadata
        doc = Document(page_content=content, metadata={"source": file_path})
        documents.append(doc)
    except Exception as e:
        logging.error(f"Error loading file '{file_path}': {e}")

    # Write removed files to log
    if removed_files:
        try:
            logs_folder = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_folder, exist_ok=True)
            removed_files_log_file = os.path.join(logs_folder, "removed_files.log")
            with open(removed_files_log_file, "w", encoding="utf-8") as log_file:
                log_file.write("Files Removed During Loading Due to Filters:\n")
                for md_file, url, reason in removed_files:
                    log_file.write(f"File: {md_file}, URL: {url}, Reason: {reason}\n")
        except Exception as e:
            logging.error(f"Error writing removed files log: {e}")

    return documents

def split_documents(documents):
    class CustomMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter):
        def __init__(self, headers_to_split_on, **kwargs):
            super().__init__(headers_to_split_on=headers_to_split_on, **kwargs)
            # Build patterns for exact header levels with unique group names
            header_patterns = []
            for header_symbol, _ in headers_to_split_on:
                level = len(header_symbol)
                pattern = rf"^(?P<header_{level}>{re.escape('#' * level)})\s+(?P<title_{level}>.*)$"
                header_patterns.append(pattern)
            # Combine the patterns
            header_pattern = "|".join(header_patterns)
            self._pattern = re.compile(header_pattern, re.MULTILINE)

        def split_text(self, text):
            matches = list(self._pattern.finditer(text))
            if not matches:
                return [Document(page_content=text, metadata={})]

            splits = []
            hierarchy = []
            for i, match in enumerate(matches):
                start_index = match.start()
                # Determine the header level and title
                for level in range(1, 7):
                    header_group = f'header_{level}'
                    title_group = f'title_{level}'
                    if match.groupdict().get(header_group):
                        header_level = level
                        title = match.group(title_group).strip()
                        break
                else:
                    header_level = None
                    title = None

                # Update hierarchy
                if header_level is not None:
                    hierarchy = hierarchy[:header_level - 1] + [title]
                else:
                    hierarchy = []

                # Determine the end of the current section
                if i + 1 < len(matches):
                    next_start = matches[i + 1].start()
                else:
                    next_start = len(text)

                content = text[start_index:next_start]
                # Include the hierarchy in the metadata
                metadata = {'hierarchy': hierarchy.copy()}
                splits.append(Document(page_content=content, metadata=metadata))

            return splits

    # Function to remove links from markdown
    def remove_links_from_markdown(text):
        # Regex pattern to match markdown links [link text](URL)
        link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        return link_pattern.sub(r'\1', text)

    # Ask the user for the language
    language = input("Enter the language of the input files (e.g., 'TypeScript', 'Python'): ").strip()

    while True:
        try:
            split_method = input("Enter split method (markdown or recursive): ").strip().lower()

            if split_method not in ["markdown", "recursive"]:
                print("Invalid choice. Please enter 'markdown' or 'recursive'.")
                continue

            remove_links = False
            if split_method == "markdown":
                # Ask if the user wants to remove links
                while True:
                    remove_links_input = input("Would you like to remove links from the markdown? (yes or no): ").strip().lower()
                    if remove_links_input in ["yes", "no"]:
                        remove_links = remove_links_input == "yes"
                        break
                    else:
                        print("Invalid choice. Please enter 'yes' or 'no'.")

                while True:
                    header_levels_input = input(
                        "Enter header levels to split on (e.g., 'All', '1', '2', or '1,2'): "
                    ).strip().lower()
                    if header_levels_input == 'all':
                        header_levels = [1, 2, 3, 4, 5, 6]
                        break
                    else:
                        try:
                            header_levels = [
                                int(h.strip()) for h in header_levels_input.split(',') if h.strip().isdigit()
                            ]
                            if not header_levels:
                                raise ValueError("No valid header levels entered.")
                            break
                        except ValueError as e:
                            print(f"Invalid input: {e}")

                headers_to_split_on = [
                    ("#" * level, f"Header {level}") for level in header_levels
                ]

                markdown_splitter = CustomMarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = []
                for doc in documents:
                    content = doc.page_content
                    # Remove links if the user opted to
                    if remove_links:
                        content = remove_links_from_markdown(content)
                    splits = markdown_splitter.split_text(content)
                    for split in splits:
                        # Combine existing metadata with new metadata
                        metadata = doc.metadata.copy()
                        metadata.update(split.metadata)
                        metadata['language'] = language
                        split.metadata = metadata
                    md_header_splits.extend(splits)
                splits = md_header_splits

            elif split_method == "recursive":
                # Ask if the user wants to remove links
                while True:
                    remove_links_input = input("Would you like to remove links from the markdown? (yes or no): ").strip().lower()
                    if remove_links_input in ["yes", "no"]:
                        remove_links = remove_links_input == "yes"
                        break
                    else:
                        print("Invalid choice. Please enter 'yes' or 'no'.")

                while True:
                    try:
                        chunk_size = int(input("Enter chunk size (positive integer): "))
                        if chunk_size <= 0:
                            raise ValueError("Chunk size must be positive.")
                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}")
                while True:
                    try:
                        chunk_overlap = int(input("Enter chunk overlap (non-negative integer): "))
                        if chunk_overlap < 0:
                            raise ValueError("Chunk overlap cannot be negative.")
                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                splits = []
                for doc in documents:
                    content = doc.page_content
                    # Remove links if the user opted to
                    if remove_links:
                        content = remove_links_from_markdown(content)
                    temp_doc = Document(page_content=content, metadata=doc.metadata)
                    doc_splits = text_splitter.split_documents([temp_doc])
                    for split in doc_splits:
                        # Add language to metadata
                        metadata = split.metadata.copy()
                        metadata['language'] = language
                        split.metadata = metadata
                    splits.extend(doc_splits)

            while True:
                preview = input("Would you like to see a preview of the split data? (yes, full, no): ").strip().lower()
                if preview in ["yes", "full", "no"]:
                    break
                else:
                    print("Invalid choice. Please enter 'yes', 'full', or 'no'.")

            if preview == "yes":
                for doc in splits[:5]:
                    print(f"Metadata: {doc.metadata}")
                    print(f"Content: {doc.page_content[:500]}...\n")
                    print("-" * 40)
            elif preview == "full":
                for doc in splits:
                    print(f"Metadata: {doc.metadata}")
                    print(f"Content: {doc.page_content}...\n")
                    print("-" * 40)

            while True:
                proceed = input("Would you like to split again with different settings or continue? (split, continue): ").strip().lower()
                if proceed in ["split", "continue"]:
                    break
                else:
                    print("Invalid choice. Please enter 'split' or 'continue'.")
            if proceed == "continue":
                break
        except Exception as e:
            logging.error(f"An error occurred while splitting documents: {e}")
            break

    return splits

def embed_and_save(splits):
    try:
        while True:
            embedding_choice = input("Choose embedding method ('huggingface' or 'ollama'): ").strip().lower()
            if embedding_choice in ['huggingface', 'ollama']:
                break
            else:
                print("Invalid choice. Please enter 'huggingface' or 'ollama'.")

        if embedding_choice == 'huggingface':
            embedding_model_name = input("Enter the embedding model to use (default: all-MiniLM-L6-v2): ") or "all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, show_progress=True)
        elif embedding_choice == 'ollama':
            model_name = input("Enter the Ollama model to use (default: llama2): ") or "nomic-embed-text"
            embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model=model_name, show_progress=True)
        else:
            # Should not reach here
            raise ValueError("Invalid embedding choice.")

        persist_directory = input("Enter the directory to save the database: ")
        os.makedirs(persist_directory, exist_ok=True)

        collection_name = input("Enter the name for the collection: ")

        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        batch_size = 5000  # Define a manageable batch size to avoid exceeding limits

        for i in range(0, len(splits), batch_size):
            batch_splits = splits[i:i + batch_size]
            # Generate UUIDs for the current batch
            uuids = [str(uuid4()) for _ in range(len(batch_splits))]

            # Ensure each document has valid metadata
            documents = []
            for split in batch_splits:
                metadata = split.metadata or {"source": "default_metadata"}
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, list):
                        # Convert list to a string
                        processed_metadata[key] = ' > '.join(map(str, value))
                    elif isinstance(value, (str, int, float, bool)):
                        processed_metadata[key] = value
                    else:
                        # Convert other types to string
                        processed_metadata[key] = str(value)
                documents.append(Document(page_content=split.page_content, metadata=processed_metadata))

            vector_store.add_documents(documents=documents, ids=uuids)

        # List collections in the database
        collections = vector_store._client.list_collections()
        print("\nDatabase Information:")
        for coll in collections:
            print(f"Collection Name: {coll.name}, Document Count: {coll.count()}")

    except Exception as e:
        logging.error(f"An error occurred during embedding and saving: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process markdown files and create embeddings.')
    parser.add_argument('--filters-off', action='store_true', help='Disable the © and 404 filters.')
    args = parser.parse_args()

    filters_off = args.filters_off

    try:
        print("Choose input method:")
        print("(1) JSON input file containing URLs and markdown data")
        print("(2) Folder of markdown files")
        print("(3) Single markdown file")

        input_choice = input("Enter 1, 2, or 3: ").strip()

        if input_choice == '1':
            # Handle JSON input file as before
            input_file = input("Enter the path of the JSON input file: ")
            if not os.path.exists(input_file):
                print("The specified input file does not exist.")
                return

            output_folder = input("Enter the path of the output folder: ")
            os.makedirs(output_folder, exist_ok=True)

            output_filename = os.path.basename(input_file).replace('.json', '')
            duplicates_log_file = f"{output_filename}-duplicates.log"

            with open(input_file, "r", encoding="utf-8") as file:
                input_data = json.load(file)

            cleaned_data, duplicates_count = clean_data(input_data, duplicates_log_file)

            save_output(cleaned_data, output_folder)

            display_summary(len(cleaned_data), duplicates_count)

            documents = load_markdown_files(output_folder, filters_off)
        elif input_choice == '2':
            # Handle folder of markdown files
            folder_path = input("Enter the path of the folder containing markdown files: ")
            if not os.path.isdir(folder_path):
                print("The specified folder does not exist.")
                return

            documents = load_markdown_files_from_folder(folder_path, filters_off)
        elif input_choice == '3':
            # Handle single markdown file
            file_path = input("Enter the path of the markdown file: ")
            if not os.path.isfile(file_path):
                print("The specified file does not exist.")
                return

            documents = load_single_markdown_file(file_path, filters_off)
        else:
            print("Invalid choice.")
            return

        split_docs = split_documents(documents)
        embed_and_save(split_docs)
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
