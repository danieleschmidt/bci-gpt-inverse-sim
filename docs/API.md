# BCI-GPT API Reference

## Authentication

All API requests require authentication using an API key.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.bci-gpt.com/api/v1/decode
```

## Endpoints

### Health Check

```http
GET /health
```

Returns system health status.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "0.1.0",
  "uptime": 3600
}
```

### EEG Decoding

```http
POST /api/v1/decode
Content-Type: application/json
```

Decode EEG signals to text.

**Request**
```json
{
  "eeg_data": [[0.1, 0.2, ...], ...],
  "sampling_rate": 1000,
  "channels": ["Fz", "Cz", "Pz", ...]
}
```

**Response**
```json
{
  "text": "hello world",
  "confidence": 0.85,
  "processing_time_ms": 45
}
```

### EEG Synthesis

```http
POST /api/v1/synthesize
Content-Type: application/json
```

Generate synthetic EEG from text.

**Request**
```json
{
  "text": "hello world",
  "duration": 2.0,
  "style": "imagined_speech"
}
```

**Response**
```json
{
  "eeg_data": [[0.1, 0.2, ...], ...],
  "realism_score": 0.92,
  "generation_time_ms": 12
}
```

## Error Handling

API uses standard HTTP status codes:

- `200` Success
- `400` Bad Request
- `401` Unauthorized
- `429` Rate Limited
- `500` Internal Server Error

**Error Response Format**
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "EEG data format is invalid",
    "details": {...}
  }
}
```

## Rate Limits

- **Standard**: 100 requests/minute
- **Premium**: 1000 requests/minute
- **Enterprise**: Custom limits

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642251600
```

## SDKs

### Python

```bash
pip install bci-gpt-client
```

```python
from bci_gpt_client import BCIGPTClient

client = BCIGPTClient(api_key="your_key")
result = client.decode_eeg(eeg_data)
print(result.text)
```

### JavaScript

```bash
npm install bci-gpt-client
```

```javascript
const { BCIGPTClient } = require('bci-gpt-client');

const client = new BCIGPTClient('your_key');
const result = await client.decodeEEG(eegData);
console.log(result.text);
```

---

*API Version: v1.0*
