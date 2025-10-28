#!/usr/bin/env python3

# Talkito - Universal TTS wrapper that works with any command
# Copyright (C) 2025 Robert Macrae
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Standalone HTTP API server for talkito - handles Claude hooks and communication webhooks."""

import json
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Callable, Optional
from dataclasses import dataclass

from .logs import log_message


@dataclass
class APIConfig:
    """Configuration for API server"""
    host: str = "0.0.0.0"
    port: int = 8080


class APIServer:
    """Standalone API server that can handle various endpoints"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.endpoints: Dict[str, Callable] = {}
        self.running = False
        
    def register_endpoint(self, path: str, handler: Callable[[dict], dict]) -> None:
        """Register a handler for a specific endpoint path - Args: path: The URL path (e.g., '/hook', '/sms'), handler: Function that takes request data dict and returns response dict"""
        self.endpoints[path.strip('/')] = handler
        log_message("INFO", f"Registered endpoint: /{path.strip('/')}")
    
    def unregister_endpoint(self, path: str) -> None:
        """Unregister a handler for a specific endpoint path"""
        path = path.strip('/')
        if path in self.endpoints:
            del self.endpoints[path]
            log_message("INFO", f"Unregistered endpoint: /{path}")
    
    def get_port(self) -> int:
        """Get the port the server is running on"""
        return self.config.port
    
    def start(self) -> bool:
        """Start the API server"""
        if self.running:
            log_message("WARNING", f"API server is already running on port {self.config.port}")
            return True
        
        # Create request handler class with access to endpoints
        parent_self = self
        
        class APIHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Override to suppress HTTP server's default logging
                pass
            
            def do_POST(self):
                """Handle POST requests"""
                # Parse the path
                parsed_path = urlparse(self.path)
                endpoint = parsed_path.path.strip('/')
                
                # Check if we have a handler for this endpoint
                if endpoint not in parent_self.endpoints:
                    self.send_error(404, "Not Found")
                    return
                
                # Read the POST data
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length).decode('utf-8')
                
                # Parse data based on content type
                content_type = self.headers.get('Content-Type', '')
                
                try:
                    if 'application/json' in content_type:
                        # JSON data
                        request_data = json.loads(post_data) if post_data else {}
                    elif 'application/x-www-form-urlencoded' in content_type:
                        # Form data (used by Twilio)
                        request_data = {k: v[0] if v else '' for k, v in parse_qs(post_data).items()}
                    else:
                        # Raw data
                        request_data = {'raw_data': post_data}
                    
                    # Add headers to request data
                    request_data['_headers'] = dict(self.headers)
                    request_data['_path'] = self.path
                    
                    # Call the handler
                    handler = parent_self.endpoints[endpoint]
                    response = handler(request_data)
                    
                    # Send response
                    status_code = response.get('status_code', 200)
                    self.send_response(status_code)
                    
                    # Set headers
                    headers = response.get('headers', {})
                    for key, value in headers.items():
                        self.send_header(key, value)
                    
                    # Default content type if not specified
                    if 'Content-Type' not in headers:
                        self.send_header('Content-Type', 'application/json')
                    
                    self.end_headers()
                    
                    # Send body
                    body = response.get('body', '')
                    if isinstance(body, dict):
                        body = json.dumps(body)
                    if isinstance(body, str):
                        body = body.encode()
                    self.wfile.write(body)
                    
                except Exception as e:
                    log_message("ERROR", f"Error handling {endpoint}: {e}")
                    self.send_error(500, "Internal Server Error")
            
            def do_GET(self):
                """Handle GET requests (for testing)"""
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                endpoints = list(parent_self.endpoints.keys())
                response = {
                    'status': 'running',
                    'endpoints': endpoints,
                    'server': 'talkito API server'
                }
                self.wfile.write(json.dumps(response).encode())
        
        # Event to signal when server is ready
        server_ready = threading.Event()
        server_error = []  # List to capture any startup errors

        # Create and start the server in a separate thread
        def run_server():
            local_server = None
            try:
                # Create server
                local_server = HTTPServer((self.config.host, self.config.port), APIHandler)
                self.server = local_server
                self.running = True
                log_message("INFO", f"HTTP API server started on {self.config.host}:{self.config.port}")

                # Signal that server is ready
                server_ready.set()

                # Start serving
                local_server.serve_forever()
            except Exception as e:
                error_msg = f"Failed to start API server: {e}"
                log_message("ERROR", error_msg)
                server_error.append(str(e))

                # Clean up on failure
                if local_server:
                    try:
                        local_server.server_close()
                    except Exception:
                        pass

                self.server = None
                self.running = False
            finally:
                # Ensure event is set even on failure
                server_ready.set()

        self.server_thread = threading.Thread(target=run_server, daemon=True, name="TalkitoAPIServer")
        self.server_thread.start()

        # Wait for server to be ready (with timeout)
        if not server_ready.wait(timeout=5.0):
            log_message("ERROR", "API server startup timeout")
            self.running = False
            return False

        # Check if there were any startup errors
        if server_error:
            log_message("ERROR", f"API server failed to start: {server_error[0]}")
            return False

        return self.running

    def stop(self):
        """Stop the API server"""
        if not self.running and not self.server:
            return  # Already stopped

        log_message("INFO", "Stopping API server")

        # First, set running to False to signal shutdown intent
        self.running = False

        # Shutdown and close the server
        if self.server:
            try:
                # Shutdown stops serve_forever()
                self.server.shutdown()
            except Exception as e:
                log_message("WARNING", f"Error during server shutdown: {e}")
            finally:
                try:
                    # server_close() releases the socket
                    self.server.server_close()
                except Exception as e:
                    log_message("WARNING", f"Error closing server socket: {e}")
                finally:
                    self.server = None

        # Wait for thread to finish
        if self.server_thread and self.server_thread.is_alive():
            try:
                self.server_thread.join(timeout=5)
                if self.server_thread.is_alive():
                    log_message("WARNING", "API server thread did not terminate within timeout")
            except Exception as e:
                log_message("WARNING", f"Error joining server thread: {e}")
            finally:
                self.server_thread = None

        log_message("INFO", "API server stopped")


# Default handlers for common endpoints
def handle_claude_hook(data: dict) -> dict:
    """Handler for Claude hooks"""
    hook_type = data.get('hook_type', 'Unknown')
    timestamp = data.get('timestamp', '')
    
    log_message("INFO", f"[CLAUDE HOOK] {hook_type} at {timestamp}")
    
    # Update state based on hook type
    from .state import get_shared_state
    state = get_shared_state()
    
    if hook_type == 'PreToolUse':
        state.set_in_tool_use(True)
        log_message("INFO", "Entering tool use mode - prompts will be spoken")
    elif hook_type == 'PostToolUse':
        state.set_in_tool_use(False)
        log_message("INFO", "Exiting tool use mode")
    
    return {
        'status_code': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': {'status': 'ok'}
    }


# Singleton instance
_api_server: Optional[APIServer] = None


def find_available_port_for_api(start_port: int = 8080, host: str = "0.0.0.0", max_attempts: int = 100) -> Optional[int]:
    """Find an available port for the API server by testing actual binding"""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Test binding to the actual host we'll use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return port
        except OSError:
            continue
    return None


def get_api_server(config: Optional[APIConfig] = None) -> APIServer:
    """Get or create the singleton API server instance"""
    global _api_server
    
    if _api_server is None:
        if config is None:
            config = APIConfig()
        _api_server = APIServer(config)
        # Register default handlers
        _api_server.register_endpoint('/hook', handle_claude_hook)
    elif config is not None and _api_server.config.port != config.port:
        # Port has changed, need to restart the server
        log_message("INFO", f"Port changed from {_api_server.config.port} to {config.port}, restarting server")
        if _api_server.running:
            _api_server.stop()
        _api_server = APIServer(config)
        # Re-register default handlers
        _api_server.register_endpoint('/hook', handle_claude_hook)
    
    return _api_server


def start_api_server(port: int = None, host: str = "0.0.0.0") -> APIServer:
    """Start the API server with the given configuration"""
    global _api_server
    
    # If there's already a running server, return it unless port changed
    if _api_server and _api_server.running:
        if port is None or _api_server.config.port == port:
            log_message("INFO", f"Using existing API server on port {_api_server.config.port}")
            return _api_server
        else:
            # Stop the existing server if port changed
            log_message("INFO", f"Stopping existing API server on port {_api_server.config.port}")
            _api_server.stop()
            _api_server = None
    
    # Find available port if not specified or if specified port is in use
    if port is None:
        port = find_available_port_for_api(8080, host)
        if port is None:
            log_message("ERROR", "Could not find an available port for API server")
            return None
        log_message("INFO", f"Found available port: {port}")
    else:
        # Test if the specified port is available
        test_port = find_available_port_for_api(port, host, max_attempts=1)
        if test_port is None:
            log_message("WARNING", f"Port {port} is in use, finding alternative")
            port = find_available_port_for_api(port + 1, host)
            if port is None:
                log_message("ERROR", "Could not find an available port for API server")
                return None
            log_message("INFO", f"Using alternative port: {port}")
    
    config = APIConfig(host=host, port=port)
    server = get_api_server(config)
    
    if not server.running:
        if not server.start():
            log_message("ERROR", f"Failed to start API server on port {port}")
            # Try to find another port
            port = find_available_port_for_api(port + 1, host)
            if port:
                log_message("INFO", f"Trying alternative port: {port}")
                config = APIConfig(host=host, port=port)
                _api_server = None  # Reset singleton
                server = get_api_server(config)
                server.start()
    
    return server