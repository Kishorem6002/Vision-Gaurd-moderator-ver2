# ==========================================================
# VISIONGUARD CONFIGURATION
# ==========================================================


# ----------------------------------------------------------
# FalconS-AI
# ----------------------------------------------------------

FALCONSAI_NSFW_THRESHOLD = 0.50


# ----------------------------------------------------------
# Person Detection
# ----------------------------------------------------------

PERSON_CONFIDENCE_THRESHOLD = 0.25

PERSON_IMAGE_SIZE = 640

PERSON_CROP_PADDING = 0.10


# ----------------------------------------------------------
# NudeNet
# ----------------------------------------------------------

NUDENET_CONFIDENCE_THRESHOLD = 0.50


# ----------------------------------------------------------
# Classes that should cause BLOCK
# ----------------------------------------------------------

DANGEROUS_NUDENET_CLASSES = {

    "FEMALE_BREAST_EXPOSED",

    "FEMALE_GENITALIA_EXPOSED",

    "MALE_GENITALIA_EXPOSED",

    "BUTTOCKS_EXPOSED",
}


# ----------------------------------------------------------
# Decision values
# ----------------------------------------------------------

DECISION_ALLOW = "ALLOW"

DECISION_BLOCK = "BLOCK"
