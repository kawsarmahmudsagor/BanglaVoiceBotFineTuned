import kagglehub

# Download latest version
path = kagglehub.competition_download('bengaliai-speech')

print("Path to competition files:", path)