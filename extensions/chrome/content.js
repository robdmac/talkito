// Content script to handle element type monitoring
class ElementLogger {
  constructor() {
    this.targetTagName = null;
    this.targetHierarchyPattern = null;
    this.observer = null;
    this.isLogging = false;
    this.lastClickedElement = null;
    this.currentMonitoredElement = null;
    this.debounceTimer = null;
    this.pendingTextContent = null;
    this.pendingReason = null;
    this._actionType = "talk"; // Default action type (private)
    this.isManualAction = false; // Track if this was a manual user action
    this.monitoringEnabled = true; // Track if monitoring is enabled via toggle
    
    // Store the last right-clicked element
    document.addEventListener('contextmenu', (e) => {
      this.lastClickedElement = e.target;
    });
    
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

      if (request.action === "startLogging") {
        this.actionType = request.actionType || "talk";
        this.isManualAction = request.isManualAction || false;
        this.startLogging(this.lastClickedElement);
      } else if (request.action === "toggleMonitoring") {
        this.monitoringEnabled = request.enabled;
        if (!request.enabled && this.isLogging) {
          this.stopLogging();
        }
      }
    });
    
    // Check monitoring state on load
    chrome.storage.local.get(['talkitoEnabled'], (result) => {
      this.monitoringEnabled = result.talkitoEnabled !== false; // Default to true
    });
    
    // Auto-activate on bolt.new
    if (window.location.hostname === 'bolt.new') {
      console.log('🌐 TalkiTo: Detected bolt.new - preparing auto-monitoring');
      this.setupBoltNewAutoMonitoring();
    }
    
    // Auto-activate on v0.dev
    if (window.location.hostname === 'v0.dev') {
      console.log('🌐 TalkiTo: Detected v0.dev - preparing auto-monitoring');
      this.setupV0DevAutoMonitoring();
    }
    
    // Auto-activate on replit.com
    if (window.location.hostname === 'replit.com' || window.location.hostname.endsWith('.replit.com')) {
      console.log('🌐 TalkiTo: Detected replit.com - preparing auto-monitoring');
      this.setupReplitAutoMonitoring();
    }
  }
  
  // Getter and setter for actionType to track changes
  get actionType() {
    return this._actionType;
  }
  
  set actionType(value) {
    this._actionType = value;
  }
  
  startLogging(element) {
    if (!element) return;
    
    // Store the action type before stopping existing logging
    const currentActionType = this.actionType;
    
    // Stop any existing logging but preserve action type
    this.stopLogging(true);
    
    // Restore the action type
    this.actionType = currentActionType;
    
    this.targetTagName = element.tagName.toLowerCase();
    this.targetHierarchyPattern = this.buildHierarchyPattern(element);
    this.isLogging = true;
    
    // Set the clicked element as the current monitored element
    this.setCurrentMonitoredElement(element, `🎯 Started monitoring ${this.targetTagName.toUpperCase()} for ${this.actionType.toUpperCase()}`);
    
    // Set up mutation observer to watch for new elements
    this.setupMutationObserver();
    
    console.log(`🎯 TalkiTo: Started monitoring <${this.targetTagName}> elements with hierarchy pattern: ${this.targetHierarchyPattern}`);
  }
  
  stopLogging(preserveActionType = false) {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    
    // Clear any pending debounce timer
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    
    // Remove visual indicator from current element
    this.removeVisualIndicator();
    
    this.isLogging = false;
    this.targetTagName = null;
    this.targetHierarchyPattern = null;
    this.currentMonitoredElement = null;
    this.pendingTextContent = null;
    this.pendingReason = null;
    
    // Only reset action type if not preserving it
    if (!preserveActionType) {
      this.actionType = "talk";
      this.isManualAction = false;
    }
  }
  
  setupBoltNewAutoMonitoring() {
    // Wait for the page to load and then find the chat messages
    const attemptAutoMonitor = () => {
      // Check if monitoring is still enabled
      if (!this.monitoringEnabled) {
        console.log('🔴 TalkiTo: Monitoring disabled, skipping auto-activation');
        return;
      }
      
      // Look for P elements in the chat area - specifically targeting assistant responses
      const selectors = [
        // Target assistant messages specifically (in flex containers, 2nd MarkdownContent)
        'div._BaseChat_1dk13_1 div._Chat_1dk13_5 div.flex div._MarkdownContent_19116_1:nth-of-type(2) p',
        'div[class*="BaseChat"] div[class*="Chat"] div.flex div[class*="MarkdownContent"]:nth-of-type(2) p',
        // Alternative selector for assistant messages
        'section.flex div.relative div.grid div.flex div._MarkdownContent_19116_1:nth-of-type(2) p',
        // Fallback to any assistant-like message structure
        'div.grid > div.flex > div[class*="MarkdownContent"] p'
      ];
      
      let targetElement = null;
      for (const selector of selectors) {
        targetElement = document.querySelector(selector);
        if (targetElement) break;
      }
      
      if (targetElement) {
        console.log('🎯 TalkiTo: Auto-activating monitoring for bolt.new assistant responses');
        console.log('📍 Found assistant message element:', targetElement);
        console.log('📝 Content preview:', targetElement.textContent.substring(0, 100) + '...');
        this.actionType = "talk";
        this.isManualAction = false;
        this.startLogging(targetElement);
      } else {
        // Retry after a bit if element not found yet
        console.log('⏳ TalkiTo: Waiting for bolt.new assistant response elements...');
        setTimeout(attemptAutoMonitor, 3000);
      }
    };
    
    // Initial delay to let page load
    setTimeout(attemptAutoMonitor, 3000);
  }
  
  setupV0DevAutoMonitoring() {
    // Wait for the page to load and then find the chat messages
    const attemptAutoMonitor = () => {
      // Check if monitoring is still enabled
      if (!this.monitoringEnabled) {
        console.log('🔴 TalkiTo: Monitoring disabled, skipping auto-activation');
        return;
      }
      
      // Look for SPAN elements in v0.dev chat area - based on the provided hierarchy
      const selectors = [
        // Specific selector based on the provided example
        'div#scroll-inner-container div.prose p span',
        // Alternative selectors for v0.dev assistant messages
        'div.group div.prose p span',
        'div[class*="prose"] p span',
        // More generic fallback
        'main div.prose span, section div.prose span'
      ];
      
      let targetElement = null;
      for (const selector of selectors) {
        targetElement = document.querySelector(selector);
        if (targetElement) break;
      }
      
      if (targetElement) {
        console.log('🎯 TalkiTo: Auto-activating monitoring for v0.dev assistant responses');
        console.log('📍 Found assistant message element:', targetElement);
        console.log('📝 Content preview:', targetElement.textContent.substring(0, 100) + '...');
        this.actionType = "talk";
        this.isManualAction = false;
        this.startLogging(targetElement);
      } else {
        // Retry after a bit if element not found yet
        console.log('⏳ TalkiTo: Waiting for v0.dev assistant response elements...');
        setTimeout(attemptAutoMonitor, 3000);
      }
    };
    
    // Initial delay to let page load
    setTimeout(attemptAutoMonitor, 3000);
  }
  
  setupReplitAutoMonitoring() {
    // Wait for the page to load and then find the chat messages
    const attemptAutoMonitor = () => {
      // Check if monitoring is still enabled
      if (!this.monitoringEnabled) {
        console.log('🔴 TalkiTo: Monitoring disabled, skipping auto-activation');
        return;
      }
      
      // Look for P elements in Replit AI chat area - specifically AI responses
      const selectors = [
        // Target AI responses specifically - they don't have span.useView_view__C2mnv in hierarchy
        'div.useView_view__C2mnv:not(:has(span.useView_view__C2mnv)) > div.rendered-markdown p',
        // Look for the second occurrence in a conversation (AI response after user message)
        'div.useView_view__C2mnv:nth-of-type(2) > div.rendered-markdown p',
        // Target messages that are likely AI responses (even-numbered children)
        'div.useView_view__C2mnv > div.useView_view__C2mnv:nth-child(even) div.rendered-markdown p',
        // More specific selector avoiding user messages
        'main#main-content div.useView_view__C2mnv > div.useView_view__C2mnv > div.useView_view__C2mnv:last-child div.rendered-markdown p',
        // Fallback that tries to avoid user input areas
        'div.rendered-markdown:not(:has(span)) p'
      ];
      
      let targetElement = null;
      for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);
        // Try to find an element that's likely an AI response
        for (const elem of elements) {
          // Check if this element's hierarchy contains span.useView_view__C2mnv
          // User messages have this span, AI responses don't
          const hasSpanAncestor = elem.closest('span.useView_view__C2mnv');
          if (!hasSpanAncestor) {
            targetElement = elem;
            break;
          }
        }
        if (targetElement) break;
      }
      
      if (targetElement) {
        console.log('🎯 TalkiTo: Auto-activating monitoring for Replit AI responses');
        console.log('📍 Found assistant message element:', targetElement);
        console.log('📝 Content preview:', targetElement.textContent.substring(0, 100) + '...');
        // Double-check we're not monitoring user input
        const hierarchyPath = this.buildElementHierarchy(targetElement);
        console.log('🔍 Hierarchy check - contains span:', hierarchyPath.includes('span.useView_view__C2mnv'));
        this.actionType = "talk";
        this.isManualAction = false;
        this.startLogging(targetElement);
      } else {
        // Retry after a bit if element not found yet
        console.log('⏳ TalkiTo: Waiting for Replit AI response elements...');
        setTimeout(attemptAutoMonitor, 3000);
      }
    };
    
    // Initial delay to let page load
    setTimeout(attemptAutoMonitor, 3000);
  }
  
  setCurrentMonitoredElement(element, reason) {
    // Remove visual indicator from previous element
    this.removeVisualIndicator();
    
    // Set new current element
    this.currentMonitoredElement = element;
    
    // Add visual indicator to new element
    this.addVisualIndicator(element);
    
    // Send the element with debouncing
    this.debouncedSendElement(element, reason);
  }
  
  addVisualIndicator(element) {
    if (element && document.contains(element)) {
      // Different colors for different action types
      let outlineColor = '#ff6b6b'; // Default red for talk
      if (this.actionType === 'whatsapp') {
        outlineColor = '#25d366'; // WhatsApp green
      } else if (this.actionType === 'slack') {
        outlineColor = '#4a154b'; // Slack purple
      }
      
      element.style.outline = `2px solid ${outlineColor}`;
      element.style.outlineOffset = '2px';
      element.setAttribute('data-element-logger', 'active');
    }
  }
  
  removeVisualIndicator() {
    if (this.currentMonitoredElement && document.contains(this.currentMonitoredElement)) {
      this.currentMonitoredElement.style.outline = '';
      this.currentMonitoredElement.style.outlineOffset = '';
      this.currentMonitoredElement.removeAttribute('data-element-logger');
    }
  }
  
  // Build a hierarchy pattern that can match similar structures
  buildHierarchyPattern(element) {
    const path = [];
    let current = element;
    
    while (current && current !== document.body && current !== document.documentElement) {
      let selector = current.tagName.toLowerCase();
      
      // Add ID if present (IDs are usually unique, so keep them specific)
      if (current.id) {
        selector += `#${current.id}`;
      }
      
      // Add first class if present (classes can help identify similar elements)
      if (current.classList.length > 0) {
        selector += `.${current.classList[0]}`;
      }
      
      path.unshift(selector);
      current = current.parentElement;
    }
    
    return path.join(' > ');
  }
  
  // Check if an element matches the target hierarchy pattern
  matchesHierarchyPattern(element) {
    if (!this.targetHierarchyPattern) return false;
    
    const elementPattern = this.buildHierarchyPattern(element);
    
    // For exact matching, we compare the patterns directly
    // This ensures only elements with the same structural path are monitored
    return elementPattern === this.targetHierarchyPattern;
  }
  
  setupMutationObserver() {
    this.observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        // Check for newly added nodes
        if (mutation.addedNodes.length > 0) {
          let latestElement = null;
          
          mutation.addedNodes.forEach(node => {
            // Check if the added node is our target type
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.tagName && node.tagName.toLowerCase() === this.targetTagName) {
                // Check if it matches our hierarchy pattern
                if (this.matchesHierarchyPattern(node)) {
                  latestElement = node;
                }
              }
              
              // Also check if any child elements match our target type and pattern
              const childElements = node.querySelectorAll ? node.querySelectorAll(this.targetTagName) : [];
              for (let i = childElements.length - 1; i >= 0; i--) {
                if (this.matchesHierarchyPattern(childElements[i])) {
                  latestElement = childElements[i];
                  break; // Take the last matching child
                }
              }
            }
          });
          
          // If we found a new element that matches our pattern, switch monitoring to it
          if (latestElement) {
            this.setCurrentMonitoredElement(latestElement, '✨ New Matching Element - Now Monitoring');
          }
        }
        
        // Check for text content changes in the current monitored element
        if (this.currentMonitoredElement && 
            (mutation.type === 'characterData' || mutation.type === 'childList')) {
          
          if (mutation.target === this.currentMonitoredElement || 
              this.currentMonitoredElement.contains(mutation.target)) {
            
            // Send the updated content with debouncing
            this.debouncedSendElement(this.currentMonitoredElement, '📝 Content Updated');
          }
        }
      });
    });
    
    // Start observing
    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
      characterDataOldValue: true
    });
  }
  
  debouncedSendElement(element, reason = '🎯 Element Found') {
    if (!element) return;
    
    // Check if element still exists in DOM
    if (!document.contains(element)) {
      return;
    }
    
    // Get current text content
    const currentTextContent = element.textContent || '';
    
    // Store the pending content and reason
    this.pendingTextContent = currentTextContent;
    this.pendingReason = reason;
    
    // Clear any existing timer
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    
    // Set new timer for 450ms
    this.debounceTimer = setTimeout(() => {
      // Check if the content is still the same after 450ms
      const finalTextContent = element.textContent || '';
      
      if (finalTextContent === this.pendingTextContent) {
        // Content hasn't changed, proceed with sending
        this.sendElement(element, this.pendingReason);
      }
      
      // Clear the timer and pending content
      this.debounceTimer = null;
      this.pendingTextContent = null;
      this.pendingReason = null;
    }, 450);
  }
  
  sendElement(element, reason = '🎯 Element Found') {
    if (!element) return;
    
    // Check if monitoring is enabled
    if (!this.monitoringEnabled) {
      return;
    }
    
    // Check if element still exists in DOM
    if (!document.contains(element)) {
      return;
    }
    
    try {
      const elementData = this.extractElementData(element, reason);

      // Send to background script to send via server
      chrome.runtime.sendMessage({
        action: "sendToServer",
        data: elementData
      });
    } catch (error) {
      if (error.message && error.message.includes('Extension context invalidated')) {
        // Extension was reloaded, stop monitoring
        console.log('🔄 TalkiTo: Extension reloaded, stopping monitoring');
        this.stopLogging();
      } else {
        console.error('❌ TalkiTo: Error sending element:', error);
      }
    }
  }
  
  // Function to build element hierarchy path (for display purposes)
  buildElementHierarchy(element) {
    const path = [];
    let current = element;
    
    while (current && current !== document.body && current !== document.documentElement) {
      let selector = current.tagName.toLowerCase();
      
      // Add ID if present
      if (current.id) {
        selector += `#${current.id}`;
      }
      
      // Add first class if present
      if (current.classList.length > 0) {
        selector += `.${current.classList[0]}`;
      }
      
      // Add position among siblings if there are multiple of same type
      const siblings = Array.from(current.parentNode?.children || [])
        .filter(sibling => sibling.tagName === current.tagName);
      
      if (siblings.length > 1) {
        const index = siblings.indexOf(current) + 1;
        selector += `:nth-of-type(${index})`;
      }
      
      path.unshift(selector);
      current = current.parentElement;
    }
    
    return path.join(' > ');
  }
  
  extractElementData(element, reason) {
    return {
      reason: reason,
      tagName: element.tagName.toLowerCase(),
      id: element.id,
      classes: Array.from(element.classList),
      textContent: element.textContent || '',
      hierarchy: this.buildElementHierarchy(element),
      hierarchyPattern: this.targetHierarchyPattern,
      timestamp: Date.now(),
      actionType: this.actionType,
      isManualAction: this.isManualAction
    };
  }
}

// Initialize the element logger
const elementLogger = new ElementLogger();

// Add keyboard shortcut to stop logging (Ctrl+Shift+L)
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'l') {
    elementLogger.stopLogging();
  }
});

// Handle page navigation/refresh
window.addEventListener('beforeunload', () => {
  elementLogger.stopLogging();
});