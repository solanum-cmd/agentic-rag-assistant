"""
CLI entry point for the Agentic RAG Assistant.

Allows testing the ingestion and retrieval pipeline from the command line.
"""

import argparse
import sys
from pathlib import Path

from app.agent import AgentRAG


def main():
    parser = argparse.ArgumentParser(description="DocMind AI - Agentic RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file")
    ingest_parser.add_argument("filepath", type=str, help="Path to the PDF file")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="The question to ask")

    # Reset command
    subparsers.add_parser("reset", help="Reset the vector store database")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    agent = AgentRAG()

    if args.command == "ingest":
        path = Path(args.filepath)
        if not path.exists():
            print(f"Error: File '{path}' does not exist.")
            sys.exit(1)
            
        print(f"Ingesting '{path}'...")
        with open(path, "rb") as f:
            chunks = agent.ingest_pdfs([f])
        print(f"Success! Created {chunks} chunks.")

    elif args.command == "ask":
        print(f"Question: {args.question}")
        print("-" * 40)
        
        try:
            # We override streaming for CLI to print directly
            result = agent.ask(args.question)
            
            print("\nAnswer:")
            for chunk in result["answer"]:
                print(chunk, end="", flush=True)
            print("\n")
            
            agent.save_ai_response("".join(result["answer"]))
            
            print("-" * 40)
            print(f"Sources used: {len(result['retrieval_result'].source_documents)}")
            
        except Exception as e:
            print(f"\nError: {e}")

    elif args.command == "reset":
        agent.reset_database()
        print("Database reset successfully.")


if __name__ == "__main__":
    main()
