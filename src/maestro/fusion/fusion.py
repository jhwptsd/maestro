# A cross-attention fusion model for music assessment that bridges the audio, visual, score, and temporal encoding models
#
# Combines outputs of audio_encoder, video_encoder, score_encoder via cross-attention, then
# temporal_model. Output goes to fusion/prediction.py. Trained via src/training/.