import labelbox

# Labelbox API Key
LBL_API_KEY = "nice api key"
# Labelbox Client
lb_client = labelbox.Client(api_key=LBL_API_KEY)

# Get Labelbox project
lb_project = lb_client.get_project("cky4nw7aaohqu0zdh6d75gobs")

# Export Video Annotations
labels = lb_project.video_label_generator()
labels = next(labels).annotations

print(labels[0])
