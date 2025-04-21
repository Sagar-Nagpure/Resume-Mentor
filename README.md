# Resume Mentor – Smart Resume Analysis App

## Features

- Upload resume in PDF format.
- Get ATS score with detailed breakdown and suggestions.
- Predict the job domain based on resume content.
- Automatically extract key technical and soft skills.
- Generate AI-powered suggestions to improve your resume.
- Compare two resumes side-by-side (text and visual format).
- Visual insights using Plotly bar charts.

## Tech Stack

- **Frontend**: Streamlit  
- **AI/NLP**: Cohere (LLM), Scikit-learn (for classification)  
- **PDF Parsing**: pdfplumber  
- **Visualization**: Plotly  
- **Deployment**: Streamlit Cloud / Local  

## Folder Structure

```
resume_mentor/
├── resume_classifier.pkl
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/resume-mentor.git
cd resume-mentor
pip install -r requirements.txt
```

## Setup Cohere API Key

### Option 1 – For Streamlit Cloud

Create a `.streamlit/secrets.toml` file:

```toml
COHERE_API_KEY = "your-cohere-api-key"
```

### Option 2 – For Local Environment

Create a `.env` file:

```
COHERE_API_KEY=your-cohere-api-key
```

## Running the App

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF resume.
2. The app extracts and displays the resume text.
3. Choose one or more actions: ATS scoring, domain prediction, skill extraction, suggestions, or comparison.
4. Get instant AI-generated results and feedback.

## Resume Comparison

- Upload a second resume.
- Options:
  - Text-based comparison with ATS aspects.
  - Graphical comparison (bar chart) using Plotly.

## Job Domain Prediction

- Uses a pre-trained ML classifier to predict the top 5 most relevant job domains.
- Example outputs:  
  - Software Engineering  
  - Data Science  
  - Marketing  
  - HR  
  - Sales  

## Why Use Resume Mentor?

- Understand how ATS systems may interpret your resume.
- Instantly get personalized improvement suggestions.
- Learn what skills your resume highlights most.
- Compare multiple versions of your resume.

## Future Improvements

- Add job listing API integration (based on resume content).
- Resume version history and improvement tracking.
- Resume export with embedded suggestions.

## Author

**Sagar-Nagpure**  
[GitHub Profile](https://github.com/Sagar-Nagpure)
```
