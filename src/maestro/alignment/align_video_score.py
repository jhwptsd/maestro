# Take in processed video and processed score and align them
#
# Uses dtw.py to match visual features (vision/) to the score timeline; usually derived from
# the audio alignment. Writes into Performance.alignment; required by bowing.py and posture.py.