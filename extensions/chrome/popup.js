// Popup script to handle toggle functionality
const toggle = document.getElementById('toggle');
const statusDiv = document.getElementById('status');

// Load current state
chrome.storage.local.get(['talkitoEnabled'], (result) => {
  const enabled = result.talkitoEnabled !== false; // Default to true
  updateUI(enabled);
});

// Handle toggle click
toggle.addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  chrome.storage.local.get(['talkitoEnabled'], (result) => {
    const currentState = result.talkitoEnabled !== false;
    const newState = !currentState;
    
    chrome.storage.local.set({ talkitoEnabled: newState }, () => {
      updateUI(newState);
      
      // Send message to content script
      chrome.tabs.sendMessage(tab.id, {
        action: 'toggleMonitoring',
        enabled: newState
      });
    });
  });
});

function updateUI(enabled) {
  toggle.classList.toggle('active', enabled);
  statusDiv.textContent = enabled ? 'Monitoring Active' : 'Monitoring Paused';
}