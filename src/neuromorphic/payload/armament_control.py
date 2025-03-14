"""
Safety-critical interface for armament payload control systems.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
import hashlib
import hmac


class SafetyState(Enum):
    LOCKED = "locked"
    ARMED = "armed"
    READY = "ready"
    FAULT = "fault"
    MAINTENANCE = "maintenance"


class AuthorizationLevel(Enum):
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@dataclass
class SafetyParameters:
    """Safety parameters for armament control."""
    max_rate: float  # Hz
    timeout: float   # seconds
    redundancy_level: int
    checksum: str


class ArmamentControl:
    """Safety-critical armament payload control interface."""
    
    def __init__(self):
        self.current_state = SafetyState.LOCKED
        self.authorized_users: Dict[str, AuthorizationLevel] = {}
        self.safety_params: Dict[str, SafetyParameters] = {}
        self.state_history: List[Tuple[datetime, SafetyState]] = []
        self.active_session: Optional[str] = None
        
        # Safety monitoring
        self.fault_counter = 0
        self.last_heartbeat = datetime.now()
        self.system_checksums: Dict[str, str] = {}
        
        # Command validation
        self.command_queue = asyncio.Queue()
        self.validation_tokens: Dict[str, str] = {}
    
    async def initialize_system(self, 
                              system_id: str, 
                              params: SafetyParameters) -> bool:
        """Initialize armament control system with safety parameters."""
        if system_id in self.safety_params:
            return False
        
        # Verify system integrity
        if not self._verify_system_integrity(system_id, params):
            return False
        
        self.safety_params[system_id] = params
        self.system_checksums[system_id] = self._calculate_checksum(params)
        return True
    
    def _verify_system_integrity(self, 
                               system_id: str, 
                               params: SafetyParameters) -> bool:
        """Verify system integrity and safety parameters."""
        if params.max_rate <= 0 or params.timeout <= 0:
            return False
        
        if params.redundancy_level < 2:  # Minimum dual redundancy
            return False
        
        # Verify checksum
        calculated_checksum = self._calculate_checksum(params)
        return calculated_checksum == params.checksum
    
    def _calculate_checksum(self, params: SafetyParameters) -> str:
        """Calculate security checksum for parameters."""
        data = f"{params.max_rate}{params.timeout}{params.redundancy_level}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def authorize_user(self,
                           user_id: str,
                           level: AuthorizationLevel,
                           credentials: str) -> bool:
        """Authorize user with specific access level."""
        if not self._validate_credentials(user_id, credentials):
            return False
        
        self.authorized_users[user_id] = level
        return True
    
    def _validate_credentials(self, user_id: str, credentials: str) -> bool:
        """Validate user credentials."""
        # Implement secure credential validation
        # This is a placeholder for actual security implementation
        return len(credentials) >= 32
    
    async def request_state_change(self,
                                 user_id: str,
                                 target_state: SafetyState) -> bool:
        """Request state change with safety checks."""
        if not self._can_change_state(user_id, target_state):
            return False
        
        # Generate validation token
        token = self._generate_validation_token(user_id, target_state)
        self.validation_tokens[user_id] = token
        
        # Queue state change request
        await self.command_queue.put({
            'user_id': user_id,
            'target_state': target_state,
            'token': token,
            'timestamp': datetime.now()
        })
        
        return True
    
    def _can_change_state(self, 
                         user_id: str, 
                         target_state: SafetyState) -> bool:
        """Check if state change is allowed."""
        if user_id not in self.authorized_users:
            return False
        
        user_level = self.authorized_users[user_id]
        current_state = self.current_state
        
        # State transition rules
        if current_state == SafetyState.FAULT:
            return (user_level == AuthorizationLevel.MAINTENANCE and 
                   target_state == SafetyState.MAINTENANCE)
        
        if target_state == SafetyState.ARMED:
            return user_level in [AuthorizationLevel.SUPERVISOR, 
                                AuthorizationLevel.EMERGENCY]
        
        if target_state == SafetyState.READY:
            return (current_state == SafetyState.ARMED and 
                   user_level != AuthorizationLevel.MAINTENANCE)
        
        return True
    
    def _generate_validation_token(self, 
                                 user_id: str, 
                                 target_state: SafetyState) -> str:
        """Generate secure validation token."""
        timestamp = datetime.now().timestamp()
        data = f"{user_id}{target_state.value}{timestamp}"
        # Use HMAC for secure token generation
        return hmac.new(b"secure_key", data.encode(), 
                       hashlib.sha256).hexdigest()
    
    async def process_commands(self):
        """Process queued commands with validation."""
        while True:
            command = await self.command_queue.get()
            
            # Validate command
            if not self._validate_command(command):
                self.fault_counter += 1
                continue
            
            # Execute state change
            success = await self._execute_state_change(command)
            if success:
                self.state_history.append(
                    (datetime.now(), command['target_state']))
            
            self.command_queue.task_done()
    
    def _validate_command(self, command: Dict[str, Any]) -> bool:
        """Validate command integrity and timing."""
        user_id = command['user_id']
        token = command['token']
        
        if user_id not in self.validation_tokens:
            return False
        
        if token != self.validation_tokens[user_id]:
            return False
        
        # Check command timing
        command_age = (datetime.now() - 
                      command['timestamp']).total_seconds()
        return command_age <= 5.0  # 5-second validity
    
    async def _execute_state_change(self, 
                                  command: Dict[str, Any]) -> bool:
        """Execute state change with safety checks."""
        target_state = command['target_state']
        
        # Perform redundant safety checks
        safety_checks = await asyncio.gather(
            self._safety_check_primary(target_state),
            self._safety_check_secondary(target_state)
        )
        
        if not all(safety_checks):
            self.fault_counter += 1
            return False
        
        # Update state
        self.current_state = target_state
        return True
    
    async def _safety_check_primary(self, 
                                  target_state: SafetyState) -> bool:
        """Primary safety check."""
        if self.fault_counter >= 3:
            return False
        
        heartbeat_age = (datetime.now() - 
                        self.last_heartbeat).total_seconds()
        return heartbeat_age <= 1.0
    
    async def _safety_check_secondary(self, 
                                    target_state: SafetyState) -> bool:
        """Secondary safety check."""
        # Verify system checksums
        for system_id, params in self.safety_params.items():
            if self._calculate_checksum(params) != self.system_checksums[system_id]:
                return False
        
        return True
    
    async def update_heartbeat(self):
        """Update system heartbeat."""
        self.last_heartbeat = datetime.now()
        
        # Reset fault counter if system is healthy
        if self.fault_counter > 0 and self._system_healthy():
            self.fault_counter -= 1
    
    def _system_healthy(self) -> bool:
        """Check overall system health."""
        return (self.fault_counter < 3 and 
                self.current_state != SafetyState.FAULT)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'state': self.current_state.value,
            'fault_counter': self.fault_counter,
            'last_heartbeat': self.last_heartbeat.timestamp(),
            'active_users': len(self.authorized_users),
            'command_queue_size': self.command_queue.qsize(),
            'state_history': [
                (ts.timestamp(), state.value)
                for ts, state in self.state_history[-10:]
            ]
        }