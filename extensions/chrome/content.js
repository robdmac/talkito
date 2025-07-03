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
      }
    });
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
    
    // Set new timer for 250ms
    this.debounceTimer = setTimeout(() => {
      // Check if the content is still the same after 250ms
      const finalTextContent = element.textContent || '';
      
      if (finalTextContent === this.pendingTextContent) {
        // Content hasn't changed, proceed with sending
        this.sendElement(element, this.pendingReason);
      }
      
      // Clear the timer and pending content
      this.debounceTimer = null;
      this.pendingTextContent = null;
      this.pendingReason = null;
    }, 250);
  }
  
  sendElement(element, reason = '🎯 Element Found') {
    if (!element) return;
    
    // Check if element still exists in DOM
    if (!document.contains(element)) {
      return;
    }
    
    const elementData = this.extractElementData(element, reason);

    // Send to background script to send via server
    chrome.runtime.sendMessage({
      action: "sendToServer",
      data: elementData
    });
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