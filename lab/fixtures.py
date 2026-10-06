"""Entirely synthetic examples for a local statement-workflow demonstration.

The procedure cards describe this software exercise, not police or legal rules.
"""

ACTORS = (
    {"id": "officer-a", "name": "Demo officer A", "role": "officer", "simulated_identity": True},
    {"id": "officer-b", "name": "Demo officer B", "role": "officer", "simulated_identity": True},
    {"id": "reviewer", "name": "Demo reviewer", "role": "reviewer", "simulated_identity": True},
)

DEMO_CASES = (
    {
        "id": "demo-case-a", "title": "Synthetic statement: missing bicycle", "owner_actor": "officer-a",
        "segments": [
            {"start": 0.0, "end": 5.4, "speaker": "Unverified speaker", "text": "I left my blue bicycle beside the library at about nine in the morning."},
            {"start": 5.4, "end": 11.2, "speaker": "Unverified speaker", "text": "When I returned at about eleven, the bicycle was no longer there."},
            {"start": 11.2, "end": 16.0, "speaker": "Unverified speaker", "text": "I did not see who removed it. I can provide a photograph of the bicycle."},
        ],
    },
    {
        "id": "demo-case-b", "title": "Synthetic statement: damaged window", "owner_actor": "officer-b",
        "segments": [
            {"start": 0.0, "end": 5.5, "speaker": "Unverified speaker", "text": "I noticed a broken window at the workshop on Cedar Lane this morning."},
            {"start": 5.5, "end": 10.8, "speaker": "Unverified speaker", "text": "I heard a loud sound at around seven but did not see what caused it."},
        ],
    },
)

PROCEDURE_CARDS = (
    {
        "id": "demo-procedure-review", "title": "Demo workflow: transcript review",
        "text": "In this software demonstration, an officer reviews every transcript segment against the recording before drafting. Saving a correction or confirming the unchanged text marks that segment reviewed. Speaker labels remain unverified unless the reviewer checks them.",
        "start": None, "end": None,
    },
    {
        "id": "demo-procedure-approval", "title": "Demo workflow: approval",
        "text": "In this software demonstration, a separate reviewer approves a current draft only after every transcript segment has been reviewed. Editing or replacing the transcript invalidates the draft and its approval. This is a simulated application workflow, not an official police procedure or legal rule.",
        "start": None, "end": None,
    },
    {
        "id": "demo-procedure-limitations", "title": "Demo workflow: evidence boundaries",
        "text": "This demonstration preserves the initial machine transcript and human corrections separately. A draft repeats reviewed testimony with timestamp references; it does not establish that the statements are true, identify a suspect, decide a legal classification, or produce a legally certified record.",
        "start": None, "end": None,
    },
)
