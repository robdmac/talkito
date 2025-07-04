// Popup script to handle toggle functionality
const toggle = document.getElementById('toggle');
const statusDiv = document.getElementById('status');
const subToggles = document.getElementById('subToggles');
const ttsToggle = document.getElementById('ttsToggle');
const whatsappToggle = document.getElementById('whatsappToggle');
const slackToggle = document.getElementById('slackToggle');

// API endpoint base URL - will try multiple ports
let apiPort = 8001;
const maxPort = 8010;

// Find the correct API port
async function findApiPort() {
  for (let port = 8001; port <= maxPort; port++) {
    try {
      const response = await fetch(`http://localhost:${port}/api/ping`, {
        method: 'GET',
        mode: 'cors'
      });
      if (response.ok) {
        apiPort = port;
        console.log(`Found TalkiTo API on port ${port}`);
        return true;
      }
    } catch (e) {
      // Continue trying next port
    }
  }
  console.error('Could not find TalkiTo API server');
  return false;
}

// Call API endpoints
async function callApi(endpoint, data = {}) {
  console.log(`[API Call] Starting request to ${endpoint} with data:`, data);
  
  try {
    const url = `http://localhost:${apiPort}${endpoint}`;
    console.log(`[API Call] Full URL: ${url}`);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      body: JSON.stringify(data)
    });
    
    console.log(`[API Call] Response status: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[API Call] Error response body:`, errorText);
      throw new Error(`API call failed: ${response.statusText} - ${errorText}`);
    }
    
    const result = await response.json();
    console.log(`[API Call] Success response:`, result);
    return result;
  } catch (error) {
    console.error(`[API Call] Error calling ${endpoint}:`, error);
    throw error;
  }
}

// Load current state
chrome.storage.local.get(['talkitoEnabled', 'ttsEnabled', 'whatsappEnabled', 'slackEnabled', 'whatsappNumber', 'slackChannel'], (result) => {
  console.log('[Initial Load] Storage state:', result);
  
  const enabled = result.talkitoEnabled !== false; // Default to true
  const ttsEnabled = result.ttsEnabled !== false; // Default to true when main toggle is on
  const whatsappEnabled = result.whatsappEnabled || false;
  const slackEnabled = result.slackEnabled || false;
  
  console.log('[Initial Load] Computed states:', { enabled, ttsEnabled, whatsappEnabled, slackEnabled });
  updateUI(enabled, ttsEnabled, whatsappEnabled, slackEnabled);
});

// Handle main toggle click
toggle.addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  chrome.storage.local.get(['talkitoEnabled', 'ttsEnabled', 'whatsappEnabled', 'slackEnabled'], async (result) => {
    const currentState = result.talkitoEnabled !== false;
    const newState = !currentState;
    
    // Find API port if not already found
    if (!await findApiPort()) {
      statusDiv.textContent = 'Error: TalkiTo server not found';
      return;
    }
    
    if (newState) {
      // Turning on - enable TTS automatically
      chrome.storage.local.set({ 
        talkitoEnabled: newState,
        ttsEnabled: true
      }, async () => {
        updateUI(newState, true, result.whatsappEnabled || false, result.slackEnabled || false);
        
        // Enable TTS via API
        try {
          await callApi('/api/tts/enable');
        } catch (error) {
          console.error('Failed to enable TTS:', error);
        }
        
        // Send message to content script
        chrome.tabs.sendMessage(tab.id, {
          action: 'toggleMonitoring',
          enabled: newState
        });
      });
    } else {
      // Turning off - disable all
      chrome.storage.local.set({ 
        talkitoEnabled: newState,
        ttsEnabled: false,
        whatsappEnabled: false,
        slackEnabled: false
      }, async () => {
        updateUI(newState, false, false, false);
        
        // Disable all via API
        try {
          await callApi('/api/tts/disable');
          if (result.whatsappEnabled) {
            await callApi('/api/whatsapp/stop');
          }
          if (result.slackEnabled) {
            await callApi('/api/slack/stop');
          }
        } catch (error) {
          console.error('Failed to disable services:', error);
        }
        
        // Send message to content script
        chrome.tabs.sendMessage(tab.id, {
          action: 'toggleMonitoring',
          enabled: newState
        });
      });
    }
  });
});

// Handle TTS toggle
ttsToggle.addEventListener('click', async () => {
  chrome.storage.local.get(['talkitoEnabled', 'ttsEnabled'], async (result) => {
    if (!result.talkitoEnabled) return; // Don't allow if main toggle is off
    
    const currentState = result.ttsEnabled !== false;
    const newState = !currentState;
    
    // Find API port if not already found
    if (!await findApiPort()) {
      statusDiv.textContent = 'Error: TalkiTo server not found';
      return;
    }
    
    chrome.storage.local.set({ ttsEnabled: newState }, async () => {
      ttsToggle.classList.toggle('active', newState);
      
      // Call API
      try {
        if (newState) {
          await callApi('/api/tts/enable');
        } else {
          await callApi('/api/tts/disable');
        }
      } catch (error) {
        console.error('Failed to toggle TTS:', error);
      }
    });
  });
});

// Handle WhatsApp toggle
whatsappToggle.addEventListener('click', async () => {
  console.log('[WhatsApp Toggle] Click event triggered');
  
  chrome.storage.local.get(['talkitoEnabled', 'whatsappEnabled', 'whatsappNumber'], async (result) => {
    console.log('[WhatsApp Toggle] Current storage state:', result);
    
    if (!result.talkitoEnabled) {
      console.log('[WhatsApp Toggle] Main toggle is off, ignoring click');
      return; // Don't allow if main toggle is off
    }
    
    const currentState = result.whatsappEnabled || false;
    const newState = !currentState;
    console.log('[WhatsApp Toggle] Current state:', currentState, 'New state:', newState);
    
    // Find API port if not already found
    if (!await findApiPort()) {
      console.error('[WhatsApp Toggle] Failed to find API port');
      statusDiv.textContent = 'Error: TalkiTo server not found';
      return;
    }
    console.log('[WhatsApp Toggle] Using API port:', apiPort);
    
    if (newState) {
      // Prompt for phone number if not set
      let phoneNumber = result.whatsappNumber;
      if (!phoneNumber) {
        console.log('[WhatsApp Toggle] No saved number, prompting user');
        phoneNumber = prompt('Enter WhatsApp phone number (e.g., +1234567890):');
        if (!phoneNumber) {
          console.log('[WhatsApp Toggle] User cancelled prompt');
          return; // User cancelled
        }
        console.log('[WhatsApp Toggle] User entered number:', phoneNumber);
      } else {
        console.log('[WhatsApp Toggle] Using saved number:', phoneNumber);
      }
      
      console.log('[WhatsApp Toggle] Saving state and number to storage');
      chrome.storage.local.set({ 
        whatsappEnabled: newState,
        whatsappNumber: phoneNumber
      }, async () => {
        console.log('[WhatsApp Toggle] Storage updated successfully');
        
        // Verify storage was updated
        chrome.storage.local.get(['whatsappEnabled'], (verifyResult) => {
          console.log('[WhatsApp Toggle] Verified storage state:', verifyResult);
        });
        
        // Update UI immediately
        console.log('[WhatsApp Toggle] Updating UI - setting toggle to active');
        whatsappToggle.classList.add('active');
        statusDiv.textContent = 'Enabling WhatsApp...';
        
        // Call API
        try {
          console.log('[WhatsApp Toggle] Calling API:', '/api/whatsapp/start', { phone_number: phoneNumber });
          const response = await callApi('/api/whatsapp/start', { phone_number: phoneNumber });
          console.log('[WhatsApp Toggle] API response:', response);
          
          // Success - update status
          statusDiv.textContent = 'WhatsApp enabled';
          
          // Ensure toggle stays active
          whatsappToggle.classList.add('active');
          console.log('[WhatsApp Toggle] Success - toggle should be green now');
          
          // Reset status after 2 seconds
          setTimeout(() => {
            statusDiv.textContent = 'Monitoring Active';
          }, 2000);
        } catch (error) {
          console.error('[WhatsApp Toggle] API call failed:', error);
          // Revert on error
          console.log('[WhatsApp Toggle] Reverting state due to error');
          chrome.storage.local.set({ whatsappEnabled: false }, () => {
            whatsappToggle.classList.remove('active');
            statusDiv.textContent = 'Error: Failed to start WhatsApp';
          });
        }
      });
    } else {
      console.log('[WhatsApp Toggle] Disabling WhatsApp mode');
      chrome.storage.local.set({ whatsappEnabled: newState }, async () => {
        console.log('[WhatsApp Toggle] Storage updated, updating UI');
        whatsappToggle.classList.toggle('active', newState);
        
        // Call API
        try {
          console.log('[WhatsApp Toggle] Calling API:', '/api/whatsapp/stop');
          const response = await callApi('/api/whatsapp/stop');
          console.log('[WhatsApp Toggle] API response:', response);
        } catch (error) {
          console.error('[WhatsApp Toggle] API call failed:', error);
        }
      });
    }
  });
});

// Handle Slack toggle
slackToggle.addEventListener('click', async () => {
  console.log('[Slack Toggle] Click event triggered');
  
  chrome.storage.local.get(['talkitoEnabled', 'slackEnabled', 'slackChannel'], async (result) => {
    console.log('[Slack Toggle] Current storage state:', result);
    
    if (!result.talkitoEnabled) {
      console.log('[Slack Toggle] Main toggle is off, ignoring click');
      return; // Don't allow if main toggle is off
    }
    
    const currentState = result.slackEnabled || false;
    const newState = !currentState;
    console.log('[Slack Toggle] Current state:', currentState, 'New state:', newState);
    
    // Find API port if not already found
    if (!await findApiPort()) {
      console.error('[Slack Toggle] Failed to find API port');
      statusDiv.textContent = 'Error: TalkiTo server not found';
      return;
    }
    console.log('[Slack Toggle] Using API port:', apiPort);
    
    if (newState) {
      // Prompt for channel if not set
      let channel = result.slackChannel;
      if (!channel) {
        console.log('[Slack Toggle] No saved channel, prompting user');
        channel = prompt('Enter Slack channel (e.g., #general):');
        if (!channel) {
          console.log('[Slack Toggle] User cancelled prompt');
          return; // User cancelled
        }
        console.log('[Slack Toggle] User entered channel:', channel);
      } else {
        console.log('[Slack Toggle] Using saved channel:', channel);
      }
      
      console.log('[Slack Toggle] Saving state and channel to storage');
      chrome.storage.local.set({ 
        slackEnabled: newState,
        slackChannel: channel
      }, async () => {
        console.log('[Slack Toggle] Storage updated successfully');
        
        // Verify storage was updated
        chrome.storage.local.get(['slackEnabled'], (verifyResult) => {
          console.log('[Slack Toggle] Verified storage state:', verifyResult);
        });
        
        // Update UI immediately
        console.log('[Slack Toggle] Updating UI - setting toggle to active');
        slackToggle.classList.add('active');
        statusDiv.textContent = 'Enabling Slack...';
        
        // Call API
        try {
          console.log('[Slack Toggle] Calling API:', '/api/slack/start', { channel: channel });
          const response = await callApi('/api/slack/start', { channel: channel });
          console.log('[Slack Toggle] API response:', response);
          
          // Success - update status
          statusDiv.textContent = 'Slack enabled';
          
          // Ensure toggle stays active
          slackToggle.classList.add('active');
          console.log('[Slack Toggle] Success - toggle should be green now');
          
          // Reset status after 2 seconds
          setTimeout(() => {
            statusDiv.textContent = 'Monitoring Active';
          }, 2000);
        } catch (error) {
          console.error('[Slack Toggle] API call failed:', error);
          // Revert on error
          console.log('[Slack Toggle] Reverting state due to error');
          chrome.storage.local.set({ slackEnabled: false }, () => {
            slackToggle.classList.remove('active');
            statusDiv.textContent = 'Error: Failed to start Slack';
          });
        }
      });
    } else {
      console.log('[Slack Toggle] Disabling Slack mode');
      chrome.storage.local.set({ slackEnabled: newState }, async () => {
        console.log('[Slack Toggle] Storage updated, updating UI');
        slackToggle.classList.toggle('active', newState);
        
        // Call API
        try {
          console.log('[Slack Toggle] Calling API:', '/api/slack/stop');
          const response = await callApi('/api/slack/stop');
          console.log('[Slack Toggle] API response:', response);
        } catch (error) {
          console.error('[Slack Toggle] API call failed:', error);
        }
      });
    }
  });
});

function updateUI(enabled, ttsEnabled, whatsappEnabled, slackEnabled) {
  console.log('[UpdateUI] Called with:', { enabled, ttsEnabled, whatsappEnabled, slackEnabled });
  
  toggle.classList.toggle('active', enabled);
  statusDiv.textContent = enabled ? 'Monitoring Active' : 'Monitoring Paused';
  
  // Show/hide sub-toggles
  if (enabled) {
    subToggles.classList.add('show');
    ttsToggle.classList.toggle('active', ttsEnabled);
    whatsappToggle.classList.toggle('active', whatsappEnabled);
    slackToggle.classList.toggle('active', slackEnabled);
    
    console.log('[UpdateUI] Toggle states after update:', {
      tts: ttsToggle.classList.contains('active'),
      whatsapp: whatsappToggle.classList.contains('active'),
      slack: slackToggle.classList.contains('active')
    });
    
    // Enable sub-toggles
    ttsToggle.classList.remove('disabled');
    whatsappToggle.classList.remove('disabled');
    slackToggle.classList.remove('disabled');
  } else {
    subToggles.classList.remove('show');
    // Disable all sub-toggles
    ttsToggle.classList.remove('active');
    whatsappToggle.classList.remove('active');
    slackToggle.classList.remove('active');
    ttsToggle.classList.add('disabled');
    whatsappToggle.classList.add('disabled');
    slackToggle.classList.add('disabled');
  }
}

// Initialize on load
window.addEventListener('load', async () => {
  await findApiPort();
});