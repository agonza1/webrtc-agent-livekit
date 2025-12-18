export const peerMetricsConfig = {
  // API key for local development (can be any string for local server)
  apiKey: process.env.NEXT_PUBLIC_PEERMETRICS_API_KEY || 'local-dev-key',
  
  // API root for local PeerMetrics server (must end with /v1)
  apiRoot: process.env.NEXT_PUBLIC_PEERMETRICS_API_ROOT || 'http://localhost:8081/v1',
  
  // Default server configuration
  defaultServer: {
    serverId: 'livekit-sfu-server',
    serverName: 'LiveKit SFU Server'
  },
  
  // Optional: Customize PeerMetrics options
  options: {
    getStatsInterval: 1000, // Get stats every 1 second
    debug: process.env.NODE_ENV === 'development',
    mockRequests: false,
    pageEvents: {
      pageVisibility: true
    }
  }
}; 