import requests

with open('sample_adi.json', 'rb') as f:
    r = requests.post(
        'http://localhost:8000/ingest',
        files={'file': ('sample_adi.json', f, 'application/json')},
        data={
            'document_id': 'doc-001',
            'tenant_id': 'acme-corp',
            'collection_id': 'benefits-2024',
            'document_type': 'insurance',
            'tags': 'insurance,benefits',
            'metadata': '{\"year\":\"2024\",\"source\":\"upload\"}'
        },
    )

print(f'Status code: {r.status_code}')
print(r.json())
