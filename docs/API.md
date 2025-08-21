# BCI-GPT API Documentation

## Overview

The BCI-GPT API provides endpoints for brain-computer interface operations, including EEG processing, model inference, and real-time decoding.

**Base URL**: `https://api.bci-gpt.com/v1`  
**Authentication**: Bearer Token  
**Content-Type**: `application/json`

## Authentication

All API requests require authentication using a bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" https://api.bci-gpt.com/v1/health
```

## Endpoints

### Health Check

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-21T03:23:00Z",
  "version": "1.0.0"
}
```

### EEG Processing

#### POST /eeg/process
Process EEG data for analysis.

**Request Body:**
```json
{
  "eeg_data": [[0.1, 0.2, ...], ...],
  "sampling_rate": 1000,
  "channels": ["Fz", "Cz", "Pz"],
  "preprocessing": {
    "bandpass": [0.5, 40],
    "artifact_removal": true
  }
}
```

**Response:**
```json
{
  "processed_data": [[0.05, 0.15, ...], ...],
  "features": {
    "alpha_power": 0.75,
    "beta_power": 0.45,
    "theta_power": 0.32
  },
  "quality_score": 0.87,
  "processing_time": 0.123
}
```

#### POST /eeg/decode
Decode EEG signals to text.

**Request Body:**
```json
{
  "eeg_data": [[0.1, 0.2, ...], ...],
  "model": "bci-gpt-v1",
  "confidence_threshold": 0.7
}
```

**Response:**
```json
{
  "decoded_text": "hello world",
  "confidence": 0.85,
  "token_probabilities": [
    {"token": "hello", "probability": 0.92},
    {"token": "world", "probability": 0.78}
  ],
  "processing_time": 0.245
}
```

### Model Management

#### GET /models
List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "bci-gpt-v1",
      "name": "BCI-GPT Base Model",
      "version": "1.0.0",
      "capabilities": ["text_decoding", "feature_extraction"],
      "languages": ["en"]
    }
  ]
}
```

#### POST /models/{model_id}/predict
Run inference with specific model.

**Parameters:**
- `model_id` (string): Model identifier

**Request Body:**
```json
{
  "input_data": [[0.1, 0.2, ...], ...],
  "options": {
    "batch_size": 32,
    "return_features": true
  }
}
```

### Real-time Streaming

#### WebSocket /stream/decode
Real-time EEG decoding stream.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.bci-gpt.com/v1/stream/decode');

// Send EEG data
ws.send(JSON.stringify({
  "eeg_chunk": [0.1, 0.2, 0.3, ...],
  "timestamp": Date.now()
}));

// Receive decoded text
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.decoded_text);
};
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "EEG data format is invalid",
    "details": {
      "field": "eeg_data",
      "expected": "array of arrays"
    }
  },
  "timestamp": "2025-08-21T03:23:00Z",
  "request_id": "req_123456"
}
```

### Error Codes
- `INVALID_INPUT` - Invalid request data
- `UNAUTHORIZED` - Authentication failed
- `RATE_LIMITED` - Too many requests
- `MODEL_NOT_FOUND` - Specified model doesn't exist
- `PROCESSING_ERROR` - Internal processing error
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable

## Rate Limiting

- **Default**: 100 requests/minute
- **Authenticated**: 1000 requests/minute
- **Enterprise**: Custom limits

Rate limit headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## SDKs

### Python SDK
```python
from bci_gpt import BCIClient

client = BCIClient(api_token="your_token")
result = client.decode_eeg(eeg_data, model="bci-gpt-v1")
print(result.decoded_text)
```

### JavaScript SDK
```javascript
import { BCIClient } from '@bci-gpt/sdk';

const client = new BCIClient({ apiToken: 'your_token' });
const result = await client.decodeEEG(eegData, { model: 'bci-gpt-v1' });
console.log(result.decodedText);
```

## Examples

### Basic EEG Processing
```python
import requests

# Process EEG data
response = requests.post(
    'https://api.bci-gpt.com/v1/eeg/process',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
    json={
        'eeg_data': eeg_samples,
        'sampling_rate': 1000,
        'channels': ['Fz', 'Cz', 'Pz']
    }
)

processed = response.json()
print(f"Quality score: {processed['quality_score']}")
```

### Real-time Decoding
```python
import asyncio
import websockets

async def decode_stream():
    uri = "wss://api.bci-gpt.com/v1/stream/decode"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # Send EEG chunk
        await websocket.send(json.dumps({
            "eeg_chunk": [0.1, 0.2, 0.3],
            "timestamp": time.time()
        }))
        
        # Receive decoded text
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Decoded: {data['decoded_text']}")

asyncio.run(decode_stream())
```

## API Versioning

- **Current Version**: v1
- **Versioning Scheme**: URL path (`/v1/`, `/v2/`)
- **Backward Compatibility**: Maintained for 12 months
- **Deprecation Notice**: 6 months advance notice

## Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limited
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service maintenance

---

For more information, see the [Developer Guide](./DEVELOPER_GUIDE.md) or contact support.
