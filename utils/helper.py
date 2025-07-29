import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Initialize once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    print("Calling preprocessing function")
    text = text.lower()
    
    # 2. Handle common resume patterns
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Keep important characters like + (C++), # (C#), . (version numbers)
    # But remove other special characters
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    
    # Handle programming languages and versions properly
    # Keep C++, C#, .NET, Python3, etc.
    text = re.sub(r'\b(\w+)\+\+\b', r'\1plusplus', text)  # C++ -> cplusplus
    text = re.sub(r'\b(\w+)#\b', r'\1sharp', text)        # C# -> csharp
    text = re.sub(r'\.net\b', 'dotnet', text)             # .NET -> dotnet
    
    # 3. Tokenize
    words = word_tokenize(text)
    
    # 4. Custom stopwords for resumes (remove less important words)
    resume_stopwords = stop_words.union({
        'experience', 'years', 'work', 'job', 'position', 'role', 'responsible', 
        'duties', 'tasks', 'skills', 'knowledge', 'ability', 'strong', 'good',
        'excellent', 'proficient', 'familiar', 'company', 'team', 'project'
    })
    
    # 5. Remove stopwords & lemmatize, but keep important technical terms
    important_terms = {
        'python', 'java', 'javascript', 'react', 'angular', 'sql', 'aws', 'azure',
        'docker', 'kubernetes', 'git', 'jira', 'agile', 'scrum', 'cplusplus', 'csharp',
        'dotnet', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'excel', 'tableau'
    }
    
    clean_words = []
    for w in words:
        if len(w) > 2:  # Keep words longer than 2 characters
            if w in important_terms or w not in resume_stopwords:
                clean_words.append(lemmatizer.lemmatize(w))
    
    # 6. Rejoin into string
    print("end of preprocessing functions")
    return " ".join(clean_words)

    

def clean_input_text(text):
    # Normalize all kinds of line breaks to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove excessive whitespace and non-breaking spaces
    text = re.sub(r'[ \xa0]+', ' ', text)
    return text.strip()

