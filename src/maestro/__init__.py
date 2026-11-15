# Main function for the app
#
# Entry point: builds a Performance (utils/Performance.py), then runs the pipeline in order:
# preprocessing -> audio/vision/score feature extraction -> alignment -> assessment (+ optional
# fusion model) -> feedback. Each stage reads from and writes to the shared Performance object.