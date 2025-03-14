"""
Remote monitoring and diagnostics for neuromorphic systems.
"""
import os
import json
import time
import asyncio
import logging
import socket
import platform
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import aiohttp
from aiohttp import web


@dataclass
class DiagnosticData:
    """Container for diagnostic data."""
    system_id: str
    timestamp: float
    metrics: Dict[str, Any]
    health_status: Dict[str, Any]
    logs: List[str]
    errors: List[Dict[str, Any]]


class RemoteDiagnostics:
    """Remote monitoring and diagnostics service."""
    
    def __init__(
        self, 
        system_core=None,
        health_monitor=None,
        base_path: str = "/Users/yessine/Oberon",
        remote_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.system_core = system_core
        self.health_monitor = health_monitor
        self.base_path = base_path
        self.remote_url = remote_url or "https://diagnostics.example.com/api"
        self.api_key = api_key
        
        # System identification
        self.system_id = self._generate_system_id()
        
        # Monitoring settings
        self.monitoring_interval = 300  # 5 minutes
        self.running = False
        self.monitor_task = None
        
        # Local diagnostics storage
        self.diagnostics_dir = os.path.join(base_path, "diagnostics")
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("remote_diagnostics")
        
        # Web server for remote access
        self.web_app = None
        self.web_runner = None
    
    def _generate_system_id(self) -> str:
        """Generate a unique system ID."""
        hostname = socket.gethostname()
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                        for elements in range(0, 48, 8)][::-1])
        return f"{hostname}-{mac}"
    
    async def start(self):
        """Start remote diagnostics service."""
        self.logger.info("Starting remote diagnostics service")
        self.running = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Start web server
        await self._start_web_server()
    
    async def stop(self):
        """Stop remote diagnostics service."""
        self.logger.info("Stopping remote diagnostics service")
        self.running = False
        
        # Stop monitoring task
        if self.monitor_task:
            try:
                self.monitor_task.cancel()
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop web server
        if self.web_runner:
            await self.web_runner.cleanup()
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.running:
                await self._collect_and_send_diagnostics()
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _collect_and_send_diagnostics(self):
        """Collect and send diagnostic data."""
        try:
            # Collect diagnostic data
            diagnostic_data = await self._collect_diagnostics()
            
            # Save locally
            self._save_diagnostics_locally(diagnostic_data)
            
            # Send to remote server if configured
            if self.remote_url and self.api_key:
                await self._send_diagnostics(diagnostic_data)
                
        except Exception as e:
            self.logger.error(f"Error collecting/sending diagnostics: {str(e)}")
    
    async def _collect_diagnostics(self) -> DiagnosticData:
        """Collect diagnostic data from system."""
        # Basic system info
        system_metrics = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "uptime": self._get_system_uptime()
        }
        
        # Get health status if available
        health_status = {}
        if self.health_monitor:
            health_status = self.health_monitor.get_system_health()
        
        # Get recent logs
        logs = self._get_recent_logs(20)  # Last 20 log entries
        
        # Get recent errors
        errors = self._get_recent_errors()
        
        return DiagnosticData(
            system_id=self.system_id,
            timestamp=time.time(),
            metrics=system_metrics,
            health_status=health_status,
            logs=logs,
            errors=errors
        )
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return uptime_seconds
        except:
            # Fallback for non-Linux systems
            return 0.0
    
    def _get_recent_logs(self, count: int = 20) -> List[str]:
        """Get recent log entries."""
        logs = []
        log_file = os.path.join(self.base_path, "logs", "system.log")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    logs = lines[-count:] if len(lines) > count else lines
            except Exception as e:
                self.logger.error(f"Error reading logs: {str(e)}")
        
        return logs
    
    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent error records."""
        errors = []
        error_file = os.path.join(self.base_path, "logs", "errors.json")
        
        if os.path.exists(error_file):
            try:
                with open(error_file, 'r') as f:
                    error_data = json.load(f)
                    errors = error_data.get("errors", [])[-10:]  # Last 10 errors
            except Exception as e:
                self.logger.error(f"Error reading error log: {str(e)}")
        
        return errors
    
    def _save_diagnostics_locally(self, data: DiagnosticData):
        """Save diagnostic data to local storage."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"diagnostics-{timestamp}.json"
        filepath = os.path.join(self.diagnostics_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "system_id": data.system_id,
                    "timestamp": data.timestamp,
                    "metrics": data.metrics,
                    "health_status": data.health_status,
                    "logs": data.logs,
                    "errors": data.errors
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving diagnostics: {str(e)}")
    
    async def _send_diagnostics(self, data: DiagnosticData):
        """Send diagnostic data to remote server."""
        if not self.remote_url or not self.api_key:
            return
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-System-ID": data.system_id
        }
        
        payload = {
            "system_id": data.system_id,
            "timestamp": data.timestamp,
            "metrics": data.metrics,
            "health_status": data.health_status,
            "logs": data.logs,
            "errors": data.errors
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.remote_url}/diagnostics", 
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Error sending diagnostics: {response.status}")
                    else:
                        self.logger.info("Diagnostics sent successfully")
        except Exception as e:
            self.logger.error(f"Error sending diagnostics: {str(e)}")
    
    async def _start_web_server(self):
        """Start web server for remote diagnostics access."""
        self.web_app = web.Application()
        self.web_app.add_routes([
            web.get('/api/status', self._handle_status),
            web.get('/api/diagnostics', self._handle_diagnostics),
            web.post('/api/collect', self._handle_collect)
        ])
        
        self.web_runner = web.AppRunner(self.web_app)
        await self.web_runner.setup()
        
        # Bind to localhost only for security
        site = web.TCPSite(self.web_runner, '127.0.0.1', 8080)
        await site.start()
        self.logger.info("Diagnostic web server started on http://127.0.0.1:8080")
    
    async def _handle_status(self, request):
        """Handle status request."""
        status = {
            "system_id": self.system_id,
            "status": "online",
            "timestamp": time.time()
        }
        
        if self.health_monitor:
            status["health"] = self.health_monitor.get_system_health()
        
        return web.json_response(status)
    
    async def _handle_diagnostics(self, request):
        """Handle diagnostics request."""
        # Check API key
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header[7:] != self.api_key:
            return web.json_response({"error": "Unauthorized"}, status=401)
        
        # Get latest diagnostic file
        files = os.listdir(self.diagnostics_dir)
        diagnostic_files = [f for f in files if f.startswith('diagnostics-')]
        
        if not diagnostic_files:
            return web.json_response({"error": "No diagnostics available"}, status=404)
        
        latest_file = sorted(diagnostic_files)[-1]
        filepath = os.path.join(self.diagnostics_dir, latest_file)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_collect(self, request):
        """Handle request to collect new diagnostics."""
        # Check API key
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header[7:] != self.api_key:
            return web.json_response({"error": "Unauthorized"}, status=401)
        
        try:
            # Collect new diagnostics
            diagnostic_data = await self._collect_diagnostics()
            self._save_diagnostics_locally(diagnostic_data)
            
            return web.json_response({
                "status": "success",
                "timestamp": time.time(),
                "message": "Diagnostics collected successfully"
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


# Add missing import at the top
import uuid