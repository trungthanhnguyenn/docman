#!/bin/bash

echo "Testing download-by-key endpoint..."

curl -X 'POST' \
  'http://localhost:8000/api/v1/documents/download-by-key/{your_session_id_here}' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
        {"category": "ami", "index": "1"},
        {"category": "bci_1", "index": "1"},
        {"category": "bci_2", "index": "1"},
        {"category": "bci_3", "index": "2"}
]'

echo -e "\n\nTest completed!"
