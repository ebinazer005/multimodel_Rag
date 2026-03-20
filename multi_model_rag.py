import os
import pytesseract

# Force PATH for tesseract
os.environ["PATH"] += r";C:\Users\Ebinazer\AppData\Local\Programs\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Ebinazer\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()






def partition_document(file_path : str):
        print(f"partitioing document : {file_path}")
        
        elements = partition_pdf(
                filename = file_path, #path for pdf
                strategy=  "hi_res", # "fast", # using for get the more accurate aunswar from the pdf
                infer_table_structure=True,  # this keep the tables in structured html for better readable      
                extract_image_block_types=["image"], #it using to extract the image from the pdf 
                extract_image_block_to_payload=True   #help to store a image to base64 in vector db
        )

        print(f"Extracted {len(elements)} elements")
        print(len(elements))
        print(elements[30])


        #extract images 

        image = [element for element in elements if element.category == 'Image']
        print(f"Images found: {len(image)}")


        #extract the table 
        table =[table for table in elements if table.category == 'Table']
        print(len(table))
        return elements



def create_chunks_by_title(elements):

        print("create a chunks")

        chunks = chunk_by_title(
                elements, # The parsed PDF elements from previous step
                max_characters=3000, # maximum length of the chunk
                new_after_n_chars=2400, # If a chunk grows beyond ~2400 characters, try to start a new chunk at the next logical break
                combine_text_under_n_chars=500 # If a chunk is smaller than 500 characters, try to merge it with nearby chunks
        )
                
        print(chunks[4].metadata.orig_elements)
        return chunks


def separate_content_types(chunk):
        print("analysis the chunks")
        content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
          # Check for tables and images from othe riginal elements     
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                for element in chunk.metadata.orig_elements:
                        element_type = type(element).__name__  
                        if element_type == 'Table':
                                content_data['types'].append('table')
                                table_html = getattr(element.metadata, 'text_as_html', element.text)  #---> convert a table to html table format
                                content_data['tables'].append(table_html)

                        elif element_type == 'Image':
                                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                                        content_data['types'].append('image')
                                        content_data['images'].append(element.metadata.image_base64)
        content_data['types'] = list(set(content_data['types']))
        return content_data     

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
        print("Create AI-enhanced summary for mixed content")

        try:
                # Initialize LLM (needs vision model for images)
                llm = ChatGroq(
                        model="llama-3.1-8b-instant",  # vision capable + free
                        api_key=os.getenv("GROQ_API_KEY")
                )
                prompt_text = f"""You are creating a searchable description for document content retrieval.
CONTENT TO ANALYZE:
TEXT CONTENT:
{text}"""

                if tables:
                        prompt_text += "TABLES:\n"
                        for i, table in enumerate(tables):
                                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                        YOUR TASK:
                        Generate a comprehensive, searchable description that covers:

                        1. Key facts, numbers, and data points from text and tables
                        2. Main topics and concepts discussed  
                        3. Questions this content could answer
                        4. Visual content analysis (charts, diagrams, patterns in images)
                        5. Alternative search terms users might use

                        Make it detailed and searchable - prioritize findability over brevity.

                        SEARCHABLE DESCRIPTION:"""
                message_content = [{"type": "text", "text": prompt_text}]
                for image_base64 in images:
                        message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        })
        
                # Send to AI and get response
                message = HumanMessage(content=message_content)
                response = llm.invoke([message])


                return  response.content  
        except Exception as e:
                print(f" AI summary failed: {e}")
        
                # Fallback to simple summary
                summary = f"{text[:300]}..."
                if tables:
                        summary += f" [Contains {len(tables)} table(s)]"
                if images:
                        summary += f" [Contains {len(images)} image(s)]"
                return summary       

def summarise_chunks(chunks):
        print("Processing chunks with AI Summaries...")
        langchain_documents = []

        total_chunk = len(chunks)

        for i, chunk in enumerate(chunks):
                current_chunk = i + 1
                print(f"total_chunk {current_chunk} / {total_chunk}")

                content_data = separate_content_types(chunk)

                print(f"     Types found: {content_data['types']}")
                print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

                # Create AI-enhanced summary if chunk has tables/images
                if content_data['tables'] or content_data['images']:
                        print(f"     → Creating AI summary for mixed content...")
                        try:
                                enhanced_content = create_ai_enhanced_summary(
                                content_data['text'],
                                content_data['tables'], 
                                content_data['images']
                                )
                                print(f"     → AI summary created successfully")
                                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")

                        except Exception as e:
                                print(f" AI summary failed: {e}")
                                enhanced_content = content_data['text']
                else:
                        print(f"     → Using raw text (no tables/images)")
                        enhanced_content = content_data['text']

                # Create LangChain Document with rich metadata
                doc = Document(
                        page_content=enhanced_content,
                        metadata={
                                "original_content": json.dumps({
                                "raw_text": content_data['text'],
                                "tables_html": content_data['tables'],
                                "images_base64": content_data['images']
                                })
                        }
                )
                langchain_documents.append(doc)

        print(f"Processed {len(langchain_documents)} chunks")

        return langchain_documents

def create_vector_store(documents,persist_directory= "db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def run_complete_ingestion_pipeline(pdf_path: str):
        file_path = "./docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
        elements = partition_document(file_path)
        chunks = create_chunks_by_title(elements)
        summarised_chunks = summarise_chunks(chunks)
        db = create_vector_store(summarised_chunks)

        return db
db = run_complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf") 

query = "How many attention heads does the Transformer use, and what is the dimension of each head? "

retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatGroq(
                        model="llama-3.1-8b-instant",  # vision capable + free
                        api_key=os.getenv("GROQ_API_KEY")
                )
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# Usage
final_answer = generate_final_answer(chunks, query)
print(final_answer)
        