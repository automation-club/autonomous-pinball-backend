import labelbox

# Labelbox API Key
LBL_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3k0MG9pNjkyZjYwMHplcGdlM3o2anpyIiwib3JnYW5pemF0aW9uSWQiOiJja3k0MG9ocXEyZjV6MHplcGVsYjk1N3YyIiwiYXBpS2V5SWQiOiJja3k3bDh5ZG5lZG1sMHpidjN0amphaDdsIiwic2VjcmV0IjoiNjIyMTQ1YTQ5ZWQyNWIxOWQwNTljYzFjYTE4YWUxNTAiLCJpYXQiOjE2NDE3NTI4MzQsImV4cCI6MjI3MjkwNDgzNH0.TuRT1LtGorHfaNSsvXA0_1TbX_ro0EY1805usl07ogo"

# Labelbox Client
lb_client = labelbox.Client(api_key=LBL_API_KEY)

# Get Labelbox project
lb_project = lb_client.get_project("cky4nw7aaohqu0zdh6d75gobs")

# Export Video Annotations
labels = lb_project.video_label_generator()
labels = next(labels).annotations

print(labels[0])