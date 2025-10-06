# PeerMetrics Integration with LiveKit

This document explains how to use PeerMetrics with your LiveKit application for WebRTC monitoring and analytics.

## What is PeerMetrics?

PeerMetrics is a WebRTC monitoring and analytics platform that helps you track the quality and performance of your WebRTC connections. It provides real-time insights into:

- Connection quality metrics
- Audio/video performance
- Network statistics
- Custom events and analytics

## Setup

### 1. Set Up Local PeerMetrics Server

The PeerMetrics server is included in the Docker Compose setup and will start automatically with your other services.

If you need to run it separately:

1. Clone the PeerMetrics server repository
2. Follow the local setup instructions
3. Configure your local server endpoint

### 2. Environment Variables

Configure your PeerMetrics settings in your environment variables:

```bash
# .env.local
NEXT_PUBLIC_PEERMETRICS_API_KEY=local-dev-key
NEXT_PUBLIC_PEERMETRICS_API_ROOT=http://localhost:8081  # Docker PeerMetrics API
```

### 3. Configuration

The PeerMetrics integration is configured in `src/config/peerMetrics.ts`:

```typescript
export const peerMetricsConfig = {
  apiKey: process.env.NEXT_PUBLIC_PEERMETRICS_API_KEY || 'local-dev-key',
  apiRoot: process.env.NEXT_PUBLIC_PEERMETRICS_API_ROOT || 'http://localhost:8081',
  
  defaultServer: {
    serverId: 'livekit-sfu-server',
    serverName: 'LiveKit SFU Server'
  },
  
  options: {
    getStatsInterval: 1000, // Get stats every 1 second
    debug: process.env.NODE_ENV === 'development',
    mockRequests: false,
    pageEvents: {
      pageVisibility: true
    }
  }
};
```

## Usage

### Basic Integration

The PeerMetrics integration is automatically initialized when a user connects to a LiveKit room. The integration is handled by the `usePeerMetrics` hook in the Playground component.

### Custom Events

You can add custom events to track specific user actions:

```typescript
import { PeerMetrics } from '@peermetrics/sdk';

// Add a custom event
await peerMetrics.addEvent({
  eventName: 'user-action'
});
```

### Mute/Unmute Events

PeerMetrics automatically tracks mute/unmute events when you call the appropriate methods:

```typescript
// When user mutes
await peerMetrics.mute();

// When user unmutes
await peerMetrics.unmute();
```

### Debug Component

A debug component is included that allows you to:
- Add custom events
- Test PeerMetrics functionality
- Monitor the integration status

The debug component appears as a floating button in the bottom-right corner when connected to a room.

## What Gets Tracked

PeerMetrics automatically tracks:

1. **Connection Quality**: RTT, packet loss, jitter
2. **Media Performance**: Audio/video bitrates, resolution, frame rates
3. **Network Statistics**: ICE connection state, candidate pairs
4. **User Events**: Mute/unmute, page visibility changes
5. **Custom Events**: Any events you add manually

## Dashboard

Once PeerMetrics is running, you can view your analytics at:
- Docker API: `http://localhost:8081` (PeerMetrics API endpoint)
- Docker Dashboard: `http://localhost:8080` (PeerMetrics web interface)

## Troubleshooting

### Common Issues

1. **Docker Services Not Running**: Make sure both PeerMetrics API and web services are running (`docker-compose ps`)
2. **API Root Not Set**: Ensure `NEXT_PUBLIC_PEERMETRICS_API_ROOT` points to `http://localhost:8081`
3. **No Data Showing**: Ensure you're connected to a LiveKit room and the room has active participants
4. **Console Errors**: Check the browser console for any PeerMetrics-related errors

### Debug Mode

Enable debug mode by setting `debug: true` in the PeerMetrics configuration. This will log detailed information to the console.

## Advanced Configuration

### Custom Server Configuration

You can customize the server information:

```typescript
usePeerMetrics(room, {
  apiKey: 'your-api-key',
  userId: 'user-123',
  conferenceId: 'room-456',
  serverId: 'my-custom-sfu',
  serverName: 'My Custom SFU Server'
});
```

### Custom Options

You can customize various PeerMetrics options:

```typescript
const peerMetrics = new PeerMetrics({
  apiKey: 'local-dev-key',
  userId: 'user-123',
  conferenceId: 'room-456',
  apiRoot: 'http://localhost:8081',
  getStatsInterval: 2000, // Get stats every 2 seconds
  debug: true,
  mockRequests: false
});
```

## Support

For more information about PeerMetrics, visit:
- [PeerMetrics GitHub Repository](https://github.com/peermetrics)
- [Local Server Setup Guide](https://github.com/peermetrics/peermetrics) 