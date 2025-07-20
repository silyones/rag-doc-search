import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRAG:
    """A Retrieval-Augmented Generation system for document search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', data_dir: str = "data"):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Name of the sentence transformer model
            data_dir: Directory containing text documents
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.model = None
        self.index = None
        self.paragraphs = []
        self.sources = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the model and build the search index."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        self._load_documents()
        self._build_index()
        
        logger.info(f"RAG system initialized with {len(self.paragraphs)} paragraphs")
    
    def _load_documents(self) -> None:
        """Load and process documents from the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found")
        
        self.paragraphs = []
        self.sources = []
        
        txt_files = list(self.data_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in '{self.data_dir}'")
        
        logger.info(f"Processing {len(txt_files)} files...")
        
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                # Split into paragraphs and filter empty ones
                paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
                
                self.paragraphs.extend(paragraphs)
                self.sources.extend([file_path.name] * len(paragraphs))
                
                logger.info(f"Loaded {len(paragraphs)} paragraphs from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def _build_index(self) -> None:
        """Build the FAISS search index from document embeddings."""
        if not self.paragraphs:
            raise ValueError("No paragraphs loaded. Cannot build index.")
        
        logger.info("Encoding paragraphs...")
        embeddings = self.model.encode(
            self.paragraphs, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Built FAISS index with dimension {dimension}")
    
    def search(self, query: str, k: int = 1) -> str:
        """
        Search for the most relevant paragraph(s) to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            Formatted string with the most relevant paragraph and its source
        """
        if not query.strip():
            return "❌ Please enter a valid question."
        
        if self.index is None:
            return "❌ Search index not initialized."
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Format result
            idx = indices[0][0]
            distance = distances[0][0]
            paragraph = self.paragraphs[idx]
            source = self.sources[idx]
            
            # Calculate similarity score (convert L2 distance to similarity)
            similarity = max(0, 1 - (distance / 2))
            
            result = f"""
## **Source: {source}**
**Similarity Score: {similarity:.2%}**

{paragraph}
            """.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"❌ Search failed: {str(e)}"


def create_interface(rag_system: DocumentRAG) -> gr.Interface:
    """Create and configure the Gradio interface."""
    
    def search_wrapper(query: str) -> str:
        """Wrapper function for Gradio interface."""
        return rag_system.search(query)
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-markdown {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        border-left: 4px solid #007bff;
        color: #212529 !important;
    }
    .output-markdown * {
        color: #212529 !important;
    }
    """
    
    interface = gr.Interface(
        fn=search_wrapper,
        inputs=gr.Textbox(
            label="❓ Ask a Question",
            placeholder="Enter your question here...",
            lines=2
        ),
        outputs=gr.Markdown(
            label="🎯 Most Relevant Answer",
            elem_classes=["output-markdown"]
        ),
        title="🔍 Mini RAG - Document Search System",
        description="""
        **Ask any question and get the most relevant paragraph from your knowledge base.**
        
        This system uses semantic search to find the best matching content from your documents.
        """,
        theme=gr.themes.Soft(),
        css=css,
        examples=[
            ["What is cybersecurity"],
            ["Common phishing tactics"],
            ["Tell me about SQL injection"],
        ],
        allow_flagging="never"
    )
    
    return interface


def main():
    """Main function to run the RAG application."""
    try:
        # Initialize RAG system
        rag = DocumentRAG()
        
        # Create and launch interface
        interface = create_interface(rag)
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. The 'data' directory exists")
        print("2. There are .txt files in the data directory")
        print("3. All required dependencies are installed")


if __name__ == "__main__":
    main()