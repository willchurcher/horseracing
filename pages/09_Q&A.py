import streamlit as st
from utils.navigation import add_navigation
from pathlib import Path

def parse_qa_markdown(filepath):
    """Parse Q&A markdown file into a list of question-answer pairs."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split content into sections based on headers
    sections = content.split('\n# ')
    
    # First section won't start with '# ', so remove empty first item if it exists
    if not sections[0].strip():
        sections = sections[1:]
    
    qa_pairs = []
    for section in sections:
        # Split each section into lines
        lines = section.strip().split('\n')
        
        # First line is the question (header)
        question = lines[0].strip()
        
        # Remaining lines form the answer, preserving line breaks for markdown
        answer = '\n'.join(lines[1:]).strip()
        
        if question and answer:  # Only add if both question and answer exist
            qa_pairs.append({
                "question": question,
                "answer": answer
            })
    
    return qa_pairs

# Page config
st.set_page_config(page_title="Q&A - Horse Racing Analytics", page_icon="❓", layout="wide")

# Title
st.title("Frequently Asked Questions")

# Load Q&A pairs from markdown file
qa_file = Path(__file__).parent.parent / "content" / "questions_and_answers.md"
qa_pairs = parse_qa_markdown(qa_file)

# Display Q&A pairs using expanders
for qa in qa_pairs:
    with st.expander(qa["question"]):
        st.markdown(qa["answer"])