
class TrainConfig:
    # train voice
    NAME = "F6"
    GENDER = "F"
    TARGET_WAV_PATH = "voices/F6.wav"
    TARGET_WAV_LANG = "en" # ["hi", "en", "ja"]
    REFERENCE_STYLE = "auto" # ["M2.json", None, "auto"]
    # training parameters
    SEED = 749
    SPEED = 1.05
    NUM_STEPS = 2000
    LEARNING_RATE = 0.0004
    VE_STEPS = 6
    SAVE_STEPS = 500
    EARLY_STOP_LOSS_THRESHOLD = 0.015
