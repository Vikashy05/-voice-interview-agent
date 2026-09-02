"""The interview script.

Questions live only here now - Postgres (when DATABASE_URL is configured)
records one table per interview run instead of holding a shared question
catalog, so there is no database lookup for the question list to fall back
from any more. See db.py for how a run's answers are stored.

Each question has a plain numeric id (1, 2, 3...) matching its position in
this list, and its text. The id is what ties an answer back to the question
it was given for - in results.json, in answers.md, and in that run's table.
"""
from __future__ import annotations

import random as _random

QUESTIONS = [
    {"id": 1, "text": "To start, could you tell me a bit about yourself and your background?"},
    {"id": 2, "text": "Let's talk modelling. What is overfitting, and how would you tell that a model is overfitting?"},
    {"id": 3, "text": "And the other side of that. What does underfitting look like, and how would you fix it?"},
    {"id": 4, "text": "How do you think about the trade-off between the two when you're tuning a model?"},
    {"id": 5, "text": "Say a dataset comes to you with a lot of missing values. How do you decide what to do with them?"},
    {"id": 6, "text": "When does imputation actually make a model worse rather than better?"},
    {"id": 7, "text": "Last one. What questions do you have for me about the role or the team?"},
]

# A few natural variations so back-to-back interviews (and repeat test runs)
# don't all open and close with the exact same line word for word. One of
# each is picked at random per process - see GREETING/SIGNOFF below - so the
# rest of the codebase can keep importing a single fixed string.
GREETING_OPTIONS = (
    "Hi there, thanks for taking the time today. I'm Jerry, and I'll be running "
    "your interview. We'll go through a few machine learning fundamentals, and it "
    "should take fifteen minutes or so. Feel free to just cut in any time if you "
    "want to add something. Ready when you are.",

    "Hello, and thanks for making time for this. I'm Jerry, and I'll be your "
    "interviewer today. We'll cover a handful of machine learning fundamentals "
    "over the next fifteen minutes or so, and you're welcome to jump in any time "
    "you'd like to add something. Let's get started whenever you're ready.",

    "Hi, good to have you here. I'm Jerry, and I'll be leading today's interview. "
    "We'll spend about fifteen minutes going through some core machine learning "
    "concepts, and feel free to interrupt any time you want to add to something. "
    "Whenever you're ready, we can begin.",
)

SIGNOFF_OPTIONS = (
    "That's everything from my side. Thanks so much for your time today, it was "
    "really good talking with you. We'll be in touch soon. Take care!",

    "That wraps up our questions. I really enjoyed this conversation, thank you "
    "for your time today. You'll hear from us soon. Take care!",

    "That's all I had for you today. Thanks so much for talking this through "
    "with me, it was a pleasure. We'll follow up with you soon. Take care!",
)

GREETING = _random.choice(GREETING_OPTIONS)
SIGNOFF = _random.choice(SIGNOFF_OPTIONS)
