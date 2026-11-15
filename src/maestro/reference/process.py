# Use already-exising processing pipeline with reference video, but leave blank spaces where pose or audio cannot properly be estimated as empty
#
# Reuses preprocessing/, audio/, vision/ on the reference, leaving gaps as empty where
# estimation fails. Result stored in Performance.reference; aligned by alignment/reference_alignment.py.