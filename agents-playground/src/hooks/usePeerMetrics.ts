import { useEffect, useRef, useState } from 'react';
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
  getStatsInterval?: number;
  debug?: boolean;
}

export interface PeerMetricsInstance {
  instance: PeerMetrics;
  isInitialized: boolean;
}

export function usePeerMetrics(room: Room | null, options: UsePeerMetricsOptions): PeerMetricsInstance | null {
  const peerMetricsRef = useRef<PeerMetrics | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  
  const { 
    apiKey, 
    userId, 
    userName, 
    conferenceId, 
    conferenceName,
    apiRoot,
    serverId = 'livekit-sfu-server', 
    serverName = 'LiveKit SFU Server', 
    enabled = true,
    getStatsInterval = 1000,
    debug = false
  } = options;

  useEffect(() => {
    if (!room || !enabled) {
      setIsInitialized(false);
      return;
    }

    // Initialize PeerMetrics
    const peerMetrics = new PeerMetrics({
      apiKey,
      userId,
      userName,
      conferenceId,
      conferenceName,
      apiRoot,
      getStatsInterval,
      debug
    });
    peerMetricsRef.current = peerMetrics;

    // Initialize and add LiveKit integration
    const initializePeerMetrics = async () => {
      try {
        console.log('🔷 PeerMetrics: Starting initialization...', {
          apiKey,
          userId,
          conferenceId,
          apiRoot,
          getStatsInterval,
          debug,
          enabled
        });
        
        await peerMetrics.initialize();
        console.log('✅ PeerMetrics: Initialized successfully');
        
        // Note: We're NOT using addSdkIntegration because it doesn't reliably detect
        // LiveKit peer connections. Instead, we'll manually add them below.
        console.log('📊 PeerMetrics: Will manually add peer connections (getStatsInterval:', getStatsInterval, 'ms)');
        
        // Log room and peer connection info
        console.log('🔍 LiveKit Room State:', {
          state: room.state,
          participants: room.remoteParticipants.size,
          localParticipant: room.localParticipant?.identity,
          // @ts-ignore - accessing internal engine for debugging
          engineConnected: room.engine?.client?.isConnected
        });
        
        // Wait a bit for peer connections to be established
        await new Promise(resolve => setTimeout(resolve, 2000));
        console.log('⏱️ Waited 2s for peer connections to establish');
        
        // Manually add peer connections to PeerMetrics
        // The LiveKit integration doesn't always auto-detect connections
        // @ts-ignore - accessing internal engine for debugging  
        const pc = room.engine?.pcManager?.publisher?.pc;
        // @ts-ignore
        const subscriberPC = room.engine?.pcManager?.subscriber?.pc;
        
        console.log('🔌 Peer Connections Found:', {
          publisher: pc ? 'YES' : 'NO',
          subscriber: subscriberPC ? 'YES' : 'NO',
          publisherState: pc?.connectionState,
          subscriberState: subscriberPC?.connectionState
        });
        
        // Manually add peer connections if they exist
        // Note: peerId should identify the remote peer (SFU server), not create new participants
        if (pc) {
          try {
            await peerMetrics.addConnection({
              pc: pc,
              peerId: serverId,  // Use same server ID for both connections
              isSfu: true  // LiveKit uses SFU architecture
            });
            console.log('✅ Manually added publisher peer connection to PeerMetrics (peerId:', serverId, ')');
          } catch (err) {
            console.error('Failed to add publisher connection:', err);
          }
        }
        
        if (subscriberPC) {
          try {
            await peerMetrics.addConnection({
              pc: subscriberPC,
              peerId: serverId,  // Use same server ID - it's the same SFU server
              isSfu: true  // LiveKit uses SFU architecture
            });
            console.log('✅ Manually added subscriber peer connection to PeerMetrics (peerId:', serverId, ')');
          } catch (err) {
            console.error('Failed to add subscriber connection:', err);
          }
        }
        
        if (!pc && !subscriberPC) {
          console.warn('⚠️ No peer connections found! PeerMetrics will not be able to track stats.');
        }
        
        // Mark as initialized AFTER everything is complete
        setIsInitialized(true);
        console.log('🎉 PeerMetrics: Ready to track events');
      } catch (error) {
        console.error('❌ Failed to initialize PeerMetrics:', error);
        setIsInitialized(false);
      }
    };

    initializePeerMetrics();

    // Cleanup function
    return () => {
      if (peerMetricsRef.current) {
        peerMetricsRef.current.endCall();
        peerMetricsRef.current = null;
      }
      setIsInitialized(false);
    };
  }, [room, apiKey, userId, userName, conferenceId, conferenceName, apiRoot, serverId, serverName, enabled, getStatsInterval, debug]);

  if (!peerMetricsRef.current) {
    return null;
  }

  return {
    instance: peerMetricsRef.current,
    isInitialized
  };
} 