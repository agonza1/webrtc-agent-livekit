import { useEffect, useState } from 'react';
import { PeerMetricsInstance } from '@/hooks/usePeerMetrics';

interface PeerMetricsDebugProps {
  peerMetrics: PeerMetricsInstance | null;
}

export function PeerMetricsDebug({ peerMetrics }: PeerMetricsDebugProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [customEventName, setCustomEventName] = useState('');

  const addCustomEvent = async () => {
    if (!peerMetrics?.isInitialized || !customEventName.trim()) return;
    
    try {
      await peerMetrics.instance.addEvent({
        eventName: customEventName
      });
      setCustomEventName('');
      console.log('Custom event added:', customEventName);
    } catch (error) {
      console.error('Failed to add custom event:', error);
    }
  };

  const handleMuteToggle = async () => {
    if (!peerMetrics?.isInitialized) return;
    
    try {
      // This would typically be called when the user actually mutes/unmutes
      // For demo purposes, we'll just add a custom event
      await peerMetrics.instance.addEvent({
        eventName: 'demo-mute-toggle'
      });
      console.log('Mute toggle event added');
    } catch (error) {
      console.error('Failed to add mute event:', error);
    }
  };

  if (!peerMetrics) {
    return null;
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg shadow-lg"
      >
        {isVisible ? 'Hide' : 'Show'} PeerMetrics Debug
      </button>
      
      {isVisible && (
        <div className="absolute bottom-12 right-0 bg-white border border-gray-300 rounded-lg shadow-xl p-4 w-80">
          <h3 className="font-semibold text-gray-800 mb-3">PeerMetrics Debug</h3>
          
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Add Custom Event
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customEventName}
                  onChange={(e) => setCustomEventName(e.target.value)}
                  placeholder="Event name"
                  className="flex-1 px-3 py-1 border border-gray-300 rounded text-sm"
                />
                <button
                  onClick={addCustomEvent}
                  className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm"
                >
                  Add
                </button>
              </div>
            </div>
            
            <div>
              <button
                onClick={handleMuteToggle}
                className="w-full bg-orange-600 hover:bg-orange-700 text-white px-3 py-2 rounded text-sm"
              >
                Demo: Add Mute Toggle Event
              </button>
            </div>
            
            <div className="text-xs text-gray-600">
              <p className="mb-2">
                <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  peerMetrics?.error 
                    ? 'bg-red-500' 
                    : peerMetrics?.isInitialized 
                    ? 'bg-green-500' 
                    : 'bg-yellow-500'
                }`}></span>
                <a 
                  href="http://localhost:8080/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 underline"
                >
                  PeerMetrics
                </a>
                {' '}is {
                  peerMetrics?.error 
                    ? 'error (see below)' 
                    : peerMetrics?.isInitialized 
                    ? 'initialized and monitoring' 
                    : 'initializing...'
                }.
              </p>
              {peerMetrics?.error && (
                <p className="text-red-600 font-medium mb-2">
                  ⚠️ Error: {peerMetrics.error.message}
                </p>
              )}
              <p>Check the console for detailed logs.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 