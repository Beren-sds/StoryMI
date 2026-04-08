"""config"""
from pathlib import Path

# Base directory configuration
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
QUESTIONNAIRE_DIR = RESULTS_DIR / "questionnaires_processed"


# File configuration
# QUESTIONNAIRE_JSON = QUESTIONNAIRE_DIR / "questionnaire_level1.json"
DSM5_DIR = DATA_DIR / "DSM5-TR"

# Model configuration
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TEMPERATURE = 1.0

# Vector store configuration
VECTOR_STORE_PATH = DATA_DIR / "vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Symptoms pool configuration
DOMAIN_POOL = [
    "Depression",
    "Anger",
    "Mania",
    "Anxiety",
    "Somatic Symptoms",
    "Suicidal Ideation",
    "Psychosis",
    "Sleep Problems",
    "Memory",
    "Repetitive Thoughts and Behaviors",
    "Dissociation",
    "Personality Functioning",
    "Substance Use"
]

# Identity types configuration
IDENTITY_TYPES = ["Adult", "Child 11-17", "Parent of Child 6-17"]

# Age range configuration
AGE_RANGES = {
    "Child 11-17": (11, 17),
    "Adult": (18, 65),
    "Parent of Child 6-17": (25, 50)
}

# domain_items map：domain -> items
DOMAIN_ITEMS = {
    "Depression": [0, 1],
    "Anger": [2, 3],
    "Mania": [4, 5],
    "Anxiety": [6, 7],
    "Somatic symptoms": [8, 9, 10],
    "Suicidal ideation": [11],
    "Psychosis": [12, 13],
    "Sleep problems": [14],
    "Memory": [15],
    "Repetitive thoughts and behaviors": [16, 17],
    "Dissociation": [18, 19],
    "Personality functioning": [20, 21],
    "Substance use": [22]
}

QUESTIONNAIRE = {
    "questions" : [
    "1. Little interest or pleasure in doing things? Score range: 0-4",
    "2. Feeling down, depressed, or hopeless? Score range: 0-4",
    "3. Feeling more irritated, grouchy, or angry than usual? Score range: 0-4",
    "4. Sleeping less than usual, but still have a lot of energy? Score range: 0-4",
    "5. Starting lots more projects than usual or doing more risky things than usual? Score range: 0-4",
    "6. Feeling nervous, anxious, frightened, worried, or on edge? Score range: 0-4",
    "7. Feeling panic or being frightened? Score range: 0-4",
    "8. Avoiding situations that make you anxious? Score range: 0-4",
    "9. Unexplained aches and pains (e.g., head, back, joints, abdomen, legs)? Score range: 0-4",
    "10. Feeling that your illnesses are not being taken seriously enough? Score range: 0-4",
    "11. Thoughts of actually hurting yourself? Score range: 0-4",
    "12. Hearing things other people couldn’t hear, such as voices even when no one was around? Score range: 0-4",
    "13. Feeling that someone could hear your thoughts, or that you could hear what another person was thinking? Score range: 0-4",
    "14. Problems with sleep that affected your sleep quality overall? Score range: 0-4",
    "15. Problems with memory (e.g., learning new information) or with location (e.g., finding your way home)? Score range: 0-4",
    "16. Unpleasant thoughts, urges, or images that repeatedly enter your mind? Score range: 0-4",
    "17. Feeling driven to perform certain behaviors or mental acts over and over again? Score range: 0-4",
    "18. Feeling detached or distant from yourself, your body, your physical surroundings, or your memories? Score range: 0-4",
    "19. Not knowing who you really are or what you want out of life? Score range: 0-4",
    "20. Not feeling close to other people or enjoying your relationships with them? Score range: 0-4",
    "21. Drinking at least 4 drinks of any kind of alcohol in a single day? Score range: 0-4",
    "22. Smoking any cigarettes, a cigar, or pipe, or using snuff or chewing tobacco? Score range: 0-4",
    "23. Using any of the following medicines ON YOUR OWN, that is, without a doctor’s prescription, in greater amounts or longer than prescribed [e.g., painkillers (like Vicodin), stimulants (like Ritalin or Adderall), sedatives or tranquilizers (like sleeping pills or Valium), or drugs like marijuana, cocaine or crack, club drugs (like ecstasy), hallucinogens (like LSD), heroin, inhalants or solvents (like glue), or methamphetamine (like speed)]? Score range: 0-4"
    ]
}

