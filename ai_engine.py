# ai_engine.py
# Author: Walter E. Ward
# Purpose: AI-powered fallacy detection module using Transformer models
# Version: 2.0 (Hybrid Reasoning Engine)

import torch
from transformers import pipeline
import spacy

# Load NLP tools
nlp = spacy.load("en_core_web_sm")

# Load Transformer model (zero-shot classification)
fallacy_labels = [
    "Ad Hominem",
    "Strawman",
    "Appeal to Authority",
    "Appeal to Emotion",
    "Slippery Slope",
    "False Dilemma",
    "Hasty Generalization",
    "Post Hoc",
    "Bandwagon",
    "No True Scotsman",
    "Red Herring",
    "Begging the Question",
    "Appeal to Ignorance",
    "False Analogy",
    "Tu Quoque",
    "Loaded Question",
    "Burden of Proof"
]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def ai_analyze_text(text: str):
    """Analyze text using AI + rule-based hybrid reasoning."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    results = []
    for sentence in sentences:
        ai_output = classifier(sentence, fallacy_labels)
        top_label = ai_output["labels"][0]
        confidence = float(ai_output["scores"][0])
        explanation = explain_fallacy(top_label)
        results.append({
            "sentence": sentence,
            "fallacy": top_label,
            "confidence": round(confidence, 3),
            "explanation": explanation
        })
    return results

def explain_fallacy(fallacy_type: str):
    """Provide a short explanation for each fallacy type."""
    explanations = {
        "Ad Hominem": "Attacking the person instead of the argument.",
        "Strawman": "Misrepresenting an argument to make it easier to attack.",
        "Appeal to Authority": "Using an authority as evidence without proof.",
        "Appeal to Emotion": "Manipulating emotions instead of reason.",
        "Slippery Slope": "Arguing one small step leads to a huge consequence.",
        "False Dilemma": "Presenting only two options when more exist.",
        "Hasty Generalization": "Drawing a conclusion from insufficient evidence.",
        "Post Hoc": "Assuming causation from correlation.",
        "Bandwagon": "Assuming something is true because many believe it.",
        "No True Scotsman": "Defining terms to exclude counterexamples.",
        "Red Herring": "Introducing irrelevant information to distract.",
        "Begging the Question": "Using the claim as its own evidence.",
        "Appeal to Ignorance": "Arguing something is true because it isn't proven false.",
        "False Analogy": "Making a weak comparison between unrelated things.",
        "Tu Quoque": "Dismissing criticism by turning it back on the accuser.",
        "Loaded Question": "Asking a question with a built-in assumption.",
        "Burden of Proof": "Demanding the other side disprove the claim."
    }
    return explanations.get(fallacy_type, "No detailed explanation available.")
