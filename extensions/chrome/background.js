// Background script to handle context menu creation and messaging
let serverPort = null;
let userPhoneNumber = null;
let userSlackChannel = null;

chrome.runtime.onInstalled.addListener(() => {
  // Create context menu items
  chrome.contextMenus.create({
    id: "talkitoParent",
    title: "TalkiTo",
    contexts: ["all"]
  });
  
  chrome.contextMenus.create({
    id: "talkitoTalk",
    parentId: "talkitoParent",
    title: "Talk",
    contexts: ["all"]
  });
  
  chrome.contextMenus.create({
    id: "talkitoWhatsApp",
    parentId: "talkitoParent",
    title: "WhatsApp",
    contexts: ["all"]
  });
  
  chrome.contextMenus.create({
    id: "talkitoSlack",
    parentId: "talkitoParent",
    title: "Slack",
    contexts: ["all"]
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId.startsWith("talkito")) {
    // Try to discover server port when user actually clicks
    if (!serverPort) {
      await discoverServerPort();
      if (!serverPort) {
        // Show user-friendly error message
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            alert('TalkiTo Server Not Found!\n\nPlease make sure your TalkiTo server is running on ports 8000-8010.\n\nCheck the console for more details.');
          }
        });
        console.error('TalkiTo server not found on any port 8000-8010. Please start your server and try again.');
        return;
      }
    }
    
    let actionType = "talk";
    
    if (info.menuItemId === "talkitoWhatsApp") {
      // Always prompt for phone number when user manually selects WhatsApp
      const phoneNumber = await promptForPhoneNumber(tab.id);
      if (!phoneNumber) return; // User cancelled
      userPhoneNumber = phoneNumber;
      actionType = "whatsapp";
    } else if (info.menuItemId === "talkitoSlack") {
      // Always prompt for Slack channel when user manually selects Slack
      const channel = await promptForSlackChannel(tab.id);
      if (!channel) return; // User cancelled
      userSlackChannel = channel;
      actionType = "slack";
    }
    
    // Send message to content script to start logging the clicked element type
    chrome.tabs.sendMessage(tab.id, {
      action: "startLogging",
      actionType: actionType,
      frameId: info.frameId,
      isManualAction: true // Flag to indicate this was a manual user action
    });
  }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.action === "sendToServer") {
    // Check if server port is available, rediscover if needed
    if (!serverPort) {
      console.log('🔍 Server port lost, rediscovering...');
      await discoverServerPort();
      if (!serverPort) {
        console.error('❌ TalkiTo server not found - skipping send');
        return;
      }
    }
    
    // For automatic element detection, check if we have saved credentials
    if (request.data.actionType === "whatsapp" && !userPhoneNumber) {
      console.log('📱 WhatsApp action detected but no phone number saved - skipping automatic send');
      return;
    }
    
    if (request.data.actionType === "slack" && !userSlackChannel) {
      console.log('💬 Slack action detected but no channel saved - skipping automatic send');
      return;
    }
    
    // Execute script in the page context to call the server
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      func: sendToServer,
      args: [request.data, serverPort, userPhoneNumber, userSlackChannel]
    });
  }
});

// Function to discover which port the server is running on
async function discoverServerPort() {
  console.log('🔍 TalkiTo: Looking for server on ports 8000-8010...');
  
  for (let port = 8000; port <= 8010; port++) {
    try {
      const response = await fetch(`http://localhost:${port}/api/ping`);
      if (response.ok) {
        serverPort = port;
        console.log(`✅ TalkiTo server found on port ${port}`);
        return port;
      }
    } catch (error) {
      // Continue to next port silently
    }
  }
  
  console.error('❌ TalkiTo server not found on any port 8000-8010');
  serverPort = null;
  return null;
}

// Function to prompt for phone number
async function promptForPhoneNumber(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: () => {
        const phoneNumber = prompt('TalkiTo WhatsApp Setup\n\nEnter your WhatsApp phone number with country code:\n(Example: +1234567890)');
        return phoneNumber?.trim();
      }
    }, (results) => {
      resolve(results && results[0] ? results[0].result : null);
    });
  });
}

// Function to prompt for Slack channel
async function promptForSlackChannel(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: () => {
        const channel = prompt('TalkiTo Slack Setup\n\nEnter Slack channel or user:\n(Examples: #general, @username)');
        return channel?.trim();
      }
    }, (results) => {
      resolve(results && results[0] ? results[0].result : null);
    });
  });
}

// Function to be injected and executed in page context
function sendToServer(data, serverPort, phoneNumber, slackChannel) {
  // Call the appropriate API based on action type
  async function callAPI(actionType, text) {
    const baseUrl = `http://localhost:${serverPort}/api`;
    let url, body;
    
    try {
      switch (actionType) {
        case 'talk':
          url = `${baseUrl}/speak`;
          body = { text: text };
          break;
          
        case 'whatsapp':
          url = `${baseUrl}/whatsapp`;
          body = { 
            message: text,
            to_number: phoneNumber,
            with_tts: true
          };
          break;
          
        case 'slack':
          url = `${baseUrl}/slack`;
          body = { 
            message: text,
            channel: slackChannel,
            with_tts: true
          };
          break;
          
        default:
          throw new Error(`Unknown action type: ${actionType}`);
      }
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const result = await response.json();
      console.log(`✅ TalkiTo ${actionType} success:`, result);
      return result;
    } catch (error) {
      console.error(`❌ Error calling TalkiTo ${actionType}:`, error);
      
      // Show user-friendly error notification
      const errorMsg = error.message.includes('Failed to fetch') 
        ? 'Server connection failed. Please check if your TalkiTo server is running.'
        : `Server error: ${error.message}`;
      
      // Fallback to console logging with error details
      console.group(`🔍 TalkiTo - ${data.tagName} (${actionType.toUpperCase()} Failed)`);
      console.log('Error:', errorMsg);
      console.log('Action Type:', actionType);
      console.log('Reason:', data.reason);
      console.log('Text Content:', data.textContent || '(empty)');
      console.log('Element Hierarchy:', data.hierarchy || 'Not available');
      console.log('Tag Name:', data.tagName);
      console.log('ID:', data.id || 'None');
      console.log('Classes:', data.classes.length > 0 ? data.classes.join(', ') : 'None');
      console.log('Timestamp:', new Date().toLocaleTimeString());
      console.groupEnd();
      
      // Show brief notification to user
      console.warn(`🚨 TalkiTo ${actionType} failed: ${errorMsg}`);
    }
  }

  // Only send if there's actual text content
  const textToSend = data.textContent?.trim();
  if (textToSend && textToSend.length > 0) {
    console.log(`🔊 ${data.reason} - Sending ${data.actionType} from ${data.hierarchy || data.tagName}: "${textToSend}"`);
    callAPI(data.actionType, textToSend);
  } else {
    console.log(`🔍 ${data.reason} - Element ${data.hierarchy || data.tagName} has no text content to send`);
  }
}