#!/bin/sh

# Update permissions
chmod +x "$0"

# Execute gui as script
panel serve --show PhAT_gui_script.py --websocket-max-message-size=104876000 --autoreload --port 5006
