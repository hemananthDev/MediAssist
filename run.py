"""
run.py
------
Single-command launcher for MediAssist Healthcare Chatbot.

Usage:
    python run.py              - Launch the chatbot (checks dependencies first)
    python run.py --ingest     - Run knowledge base ingestion only
    python run.py --setup      - Install dependencies only
    python run.py --help       - Show this help message
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"
ENV_FILE = BASE_DIR / ".env"


def print_banner():
    """Display the MediAssist welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║               🏥  MediAssist Healthcare Chatbot  🏥               ║
║                                                                   ║
║          AI-powered healthcare information assistant              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Ensure Python 3.10+ is being used."""
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required.")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import streamlit
        import ollama
        import chromadb
        import langchain
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependencies: {e}")
        return False


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n📦 Installing dependencies from requirements.txt...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
            check=True,
        )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_ollama():
    """Check if Ollama is running and nomic-embed-text is available."""
    print("\n🤖 Checking Ollama...")
    try:
        import ollama

        # ollama.list() returns a ListResponse object; models is a list of Model objects
        # Support both dict-style (old) and attribute-style (new) access
        response = ollama.list()
        models_raw = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])

        model_names = []
        for m in models_raw:
            if isinstance(m, dict):
                model_names.append(m.get("name", "") or m.get("model", ""))
            else:
                # Attribute access for Model objects (ollama >= 0.4)
                name = getattr(m, "model", None) or getattr(m, "name", None) or str(m)
                model_names.append(name)

        names_str = " ".join(model_names).lower()

        if "nomic-embed-text" in names_str:
            print("✓ Ollama is running and nomic-embed-text is available")
            return True
        else:
            print("⚠️  Ollama is running but nomic-embed-text model not found")
            if model_names:
                print(f"   Available models: {', '.join(model_names)}")
            print("   Run: ollama pull nomic-embed-text")
            return False

    except Exception as e:
        print(f"❌ Ollama is not running or unreachable: {e}")
        print("\n   Make sure Ollama is running:")
        print("   1. Download from: https://ollama.com/download")
        print("   2. Run: ollama serve")
        print("   3. Run: ollama pull nomic-embed-text")
        return False


def check_env_file():
    """Check if .env file exists with GROQ_API_KEY."""
    if not ENV_FILE.exists():
        print("\n⚠️  .env file not found")
        print("   1. Copy .env.example to .env")
        print("   2. Add your GROQ_API_KEY from https://console.groq.com")
        return False
    
    with open(ENV_FILE, "r") as f:
        content = f.read()
        if "GROQ_API_KEY" not in content or "your_groq_api_key_here" in content:
            print("\n⚠️  GROQ_API_KEY not configured in .env file")
            print("   Get your free API key from: https://console.groq.com")
            return False
    
    print("✓ .env file configured")
    return True


def check_vectorstore():
    """Check if the vectorstore has been built."""
    if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
        print("\n⚠️  Knowledge base not found")
        print("   Run: python run.py --ingest")
        return False
    print("✓ Knowledge base ready")
    return True


def run_ingestion():
    """Run the knowledge base ingestion script."""
    print("\n📚 Building knowledge base...")
    print("   This may take 10-15 minutes for the full PDF collection.\n")
    try:
        subprocess.run([sys.executable, "ingest.py"], cwd=str(BASE_DIR), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ingestion failed: {e}")
        return False


def launch_streamlit():
    """Launch the Streamlit app."""
    print("\n🚀 Launching MediAssist...")
    print("   The app will open in your browser at http://localhost:8501\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            cwd=str(BASE_DIR),
        )
    except KeyboardInterrupt:
        print("\n\n👋 MediAssist stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MediAssist Healthcare Chatbot Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Install dependencies only",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run knowledge base ingestion only",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip all pre-flight checks and launch directly",
    )
    
    args = parser.parse_args()

    print_banner()
    check_python_version()

    # Setup mode: install dependencies only
    if args.setup:
        install_dependencies()
        print("\n✓ Setup complete!")
        print("   Next steps:")
        print("   1. Ensure Ollama is running: ollama serve")
        print("   2. Pull the embedding model: ollama pull nomic-embed-text")
        print("   3. Configure .env with your GROQ_API_KEY")
        print("   4. Build knowledge base: python run.py --ingest")
        print("   5. Launch the app: python run.py")
        return

    # Ingest mode: build knowledge base only
    if args.ingest:
        if not check_dependencies():
            print("\n❌ Dependencies missing. Run: python run.py --setup")
            sys.exit(1)
        if not check_ollama():
            sys.exit(1)
        run_ingestion()
        print("\n✓ Knowledge base built successfully!")
        print("   Launch the app with: python run.py")
        return

    # Normal mode: pre-flight checks then launch
    if not args.skip_checks:
        print("\n🔍 Running pre-flight checks...\n")
        
        checks_passed = True
        
        if not check_dependencies():
            print("   → Run: python run.py --setup")
            checks_passed = False
        
        if not check_ollama():
            checks_passed = False
        
        if not check_env_file():
            checks_passed = False
        
        if not check_vectorstore():
            print("   → Run: python run.py --ingest")
            checks_passed = False
        
        if not checks_passed:
            print("\n❌ Pre-flight checks failed. Please resolve the issues above.")
            sys.exit(1)
        
        print("\n✓ All checks passed!")
    
    launch_streamlit()


if __name__ == "__main__":
    main()
