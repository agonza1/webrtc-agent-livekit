import { useEffect, useRef } from 'react';
import { Room } from 'livekit-client';
import { PeerMetrics } from '@peermetrics/sdk';

interface UsePeerMetricsOptions {
  apiKey: string;
  userId: string;
  userName?: string;
  conferenceId: string;
  conferenceName?: string;
  apiRoot?: string;
  serverId?: string;
  serverName?: string;
  enabled?: boolean;
}

export function usePeerMetrics(room: Room | null, options: UsePeerMetricsOptions) {
  const peerMetricsRef = useRef<PeerMetrics | null>(null);
  const { 
    apiKey, 
    userId, 
    userName, 
    conferenceId, 
    conferenceName,
    apiRoot,
    serverId = 'livekit-sfu-server', 
    serverName = 'LiveKit SFU Server', 
    enabled = true 
  } = options;

  useEffect(() => {
    if (!room || !enabled) {
      return;
    }

    // Initialize PeerMetrics
    const peerMetrics = new PeerMetrics({
      apiKey,
      userId,
      userName,
      conferenceId,
      conferenceName,
      apiRoot
    });
    peerMetricsRef.current = peerMetrics;

    // Initialize and add LiveKit integration
    const initializePeerMetrics = async () => {
      try {
        await peerMetrics.initialize();
        await peerMetrics.addSdkIntegration({
          livekit: {
            room: room,
            serverId: serverId,
            serverName: serverName
          }
        });
      } catch (error) {
        console.error('Failed to initialize PeerMetrics:', error);
      }
    };

    initializePeerMetrics();

    // Cleanup function
    return () => {
      if (peerMetricsRef.current) {
        peerMetricsRef.current.endCall();
        peerMetricsRef.current = null;
      }
    };
  }, [room, apiKey, userId, userName, conferenceId, conferenceName, apiRoot, serverId, serverName, enabled]);

  return peerMetricsRef.current;
} 