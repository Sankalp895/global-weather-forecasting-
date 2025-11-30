import os
print("Current dir:", os.getcwd())
print("Results folder exists?", os.path.exists('results'))
print("Files in results:", os.listdir('results'))
