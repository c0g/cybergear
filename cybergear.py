import atexit
import logging
import math
import struct
from dataclasses import dataclass
from enum import Flag, IntEnum, auto

import can

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorFault(Flag):
    NO_FAULT = auto()
    NOT_CALIBRATED = auto()
    HALL_ENCODING_FAILURE = auto()
    MAGNETIC_ENCODING_FAILURE = auto()
    OVER_TEMPERATURE = auto()
    OVERCURRENT = auto()
    UNDERVOLTAGE = auto()


class MotorMode(IntEnum):
    RESET = 0
    CALIBRATION = 1
    RUN = 2


@dataclass
class MotorState:
    angle: float
    velocity: float
    torque: float
    temperature: float
    faults: MotorFault
    mode: MotorMode


class ControlMode(IntEnum):
    OPERATION = 0x00
    POSITON = auto()
    SPEED = auto()
    CURRENT = auto()


class CommsType(IntEnum):
    GET_ID = 0x00
    MOTOR_CONTROL = 0x01
    MOTOR_FEEDBACK = 2
    MOTOR_ENABLE = 3
    MOTOR_DISABLE = 4
    SET_MECHANICAL_ZERO = 5
    SET_CAN_ID = 7
    READ_SINGLE_PARAMETER = 17
    WRITE_SINGLE_PARAMETER = 18
    FEEDBACK_FAULT = 21
    MODIFY_BAUD_RATE = 22


class ParameterIndex(IntEnum):
    RunMode = 0x7005
    IqRef = 0x7006
    SpdRef = 0x700A
    LimitTorque = 0x700B
    CurKp = 0x7010
    CurKi = 0x7011
    CurFiltGain = 0x7014
    LocRef = 0x7016
    LimitSpd = 0x7017
    LimitCur = 0x7018
    MechPos = 0x7019
    Iqf = 0x701A
    MechVel = 0x701B
    VBUS = 0x701C
    Rotation = 0x701D
    LocKp = 0x701E
    SpdKp = 0x701F
    SpdKi = 0x7020


@dataclass
class CybergearComm:
    comms_type: CommsType
    host_id: int
    motor_id: int
    data: list[int]
    upper_id_bits: int = 0x00

    def to_message(self):
        # This is 1 byte, need the bottom five bits, shifted so they start at bit index 28
        extended_id = (self.comms_type & 0x1F) << 24
        extended_id |= (self.upper_id_bits & 0xFF) << 16
        extended_id |= self.host_id << 8
        extended_id |= self.motor_id
        return can.Message(
            arbitration_id=extended_id, data=self.data, is_extended_id=True
        )


class CyberGearMotor:
    def __init__(
        self,
        bus,
        can_id,
        disable_on_exit=True,
        timeout=0.01,
        max_torque=None,
        max_current=None,
        max_speed=None,
    ):
        self.can_id = can_id
        self.bus = bus
        self.timeout = timeout
        self.max_torque = max_torque
        self.max_current = max_current
        self.max_speed = max_speed
        if disable_on_exit:
            atexit.register(self.disable)

    def enable(self):
        response = self._send_command(CommsType.MOTOR_ENABLE)
        if response:
            state = self._interpret_feedback(response)
            if (
                state.mode == MotorMode.RUN
                and state.faults == MotorFault.NO_FAULT
            ):
                logger.info(f"Motor {self.can_id} enabled successfully.")
                if self.max_torque is not None:
                    self.set_max_torque(self.max_torque)
                if self.max_current is not None:
                    self.set_max_current(self.max_current)
                if self.max_speed is not None:
                    self.set_max_speed(self.max_speed)
            else:
                logger.warning(
                    f"Motor {self.can_id} enable command sent, but motor is in {state.mode} mode with faults: {state.faults}"
                )
        else:
            logger.error(f"No response received when enabling motor {self.can_id}")

    def disable(self, clear_faults=False):
        if clear_faults:
            data = [1, 0, 0, 0, 0, 0, 0, 0]  # Set first byte to 1 to clear faults
        else:
            data = [0] * 8  # Default data with all zeros

        response = self._send_command(CommsType.MOTOR_DISABLE, data)
        if response:
            state = self._interpret_feedback(response)
            if state.mode == MotorMode.RESET:
                logger.info(f"Motor {self.can_id} disabled successfully.")
                if clear_faults:
                    logger.info(f"Faults cleared for motor {self.can_id}.")
            else:
                logger.warning(
                    f"Motor {self.can_id} disable command sent, but motor is in {state.mode} mode"
                )
        else:
            logger.error(f"No response received when disabling motor {self.can_id}")

    def set_angle(self, angle):
        return self._set_parameter(ParameterIndex.LocRef, angle)

    def set_velocity(self, velocity):
        return self._set_parameter(ParameterIndex.SpdRef, velocity)

    def current(self, current):
        return self._set_parameter(ParameterIndex.IqRef, current)

    def set_max_current(self, current):
        return self._set_parameter(ParameterIndex.LimitCur, current)

    def set_max_torque(self, torque):
        self.max_torque = torque
        return self._set_parameter(ParameterIndex.LimitTorque, torque)

    def set_max_speed(self, speed):
        return self._set_parameter(ParameterIndex.LimitSpd, speed)

    def set_run_mode(self, mode):
        return self._set_int_parameter(ParameterIndex.RunMode, mode)

    def update_state(self):
        response = self._send_command(CommsType.MOTOR_FEEDBACK)
        if response:
            state = self._interpret_feedback(response)
        else:
            raise TimeoutError(f"No response from motor {self.can_id}")
        return state

    def reset_error(self):
        self._send_command(CommsType.MOTOR_DISABLE, [1, 0, 0, 0, 0, 0, 0, 0])
        logger.info(f"Reset error for motor {self.can_id}")

    def _set_parameter(self, param, value):
        data = struct.pack("<HHf", param, 0, value)
        response = self._send_command(CommsType.WRITE_SINGLE_PARAMETER, data)
        interpreted = self._interpret_feedback(response)
        if interpreted.faults != MotorFault.NO_FAULT:
            self.disable()
            raise RuntimeError()
        return interpreted

    def _set_int_parameter(self, param, value):
        data = struct.pack("<HHI", param, 0, value)
        response = self._send_command(CommsType.WRITE_SINGLE_PARAMETER, data)
        interpreted = self._interpret_feedback(response)
        if interpreted.faults != MotorFault.NO_FAULT:
            self.disable()
            raise RuntimeError()
        return interpreted

    def set_can_id(self, new_id):
        response = self._send_command(
            comms_type=CommsType.SET_CAN_ID,
            data=[0x00] * 7 + [0x01],
            upper_id_bits=new_id,
        )
        mcu_id, motor_new_can_id = self._interpret_broadcast(response)
        if motor_new_can_id != new_id:
            raise RuntimeError(f"Setting new can ID {new_id} on {self.can_id} failed!")
        self.can_id = motor_new_can_id
        return mcu_id, motor_new_can_id

    def get_mcu_id(self):
        response = self._send_command(CommsType.GET_ID)
        mcu_id, _ = self._interpret_broadcast(response)
        return bytes(mcu_id)

    def _send_command(self, comms_type, data=None, upper_id_bits=0x00):
        if data is None:
            data = [0] * 8
        message = CybergearComm(
            comms_type=comms_type,
            host_id=0x00,
            motor_id=self.can_id,
            data=data,
            upper_id_bits=upper_id_bits,
        )
        self.bus.send(message.to_message())
        response = self.bus.recv(self.timeout)
        if response is None:
            raise TimeoutError(f"No response from motor {self.can_id}")
        return response

    @staticmethod
    def _decode_extended_id(extended_id):
        motor_can_id = (extended_id >> 8) & 0xFF
        fault_bits = (extended_id >> 16) & 0x3F
        mode_bits = (extended_id >> 22) & 0x03
        host_can_id = extended_id & 0xFF

        faults = MotorFault.NO_FAULT
        if fault_bits & 0x20:
            faults |= MotorFault.NOT_CALIBRATED
        if fault_bits & 0x10:
            faults |= MotorFault.HALL_ENCODING_FAILURE
        if fault_bits & 0x08:
            faults |= MotorFault.MAGNETIC_ENCODING_FAILURE
        if fault_bits & 0x04:
            faults |= MotorFault.OVER_TEMPERATURE
        if fault_bits & 0x02:
            faults |= MotorFault.OVERCURRENT
        if fault_bits & 0x01:
            faults |= MotorFault.UNDERVOLTAGE

        mode = MotorMode(mode_bits)

        return host_can_id, motor_can_id, faults, mode

    def _interpret_feedback(self, msg):
        _, _, faults, mode = self._decode_extended_id(msg.arbitration_id)
        angle, velocity, torque, temperature = self._decode_data(msg.data)
        return MotorState(angle, velocity, torque, temperature, faults, mode)

    def _interpret_broadcast(self, msg):
        mcu_id = msg.data
        motor_can_id = (msg.arbitration_id >> 8) & 0xFF
        return mcu_id, motor_can_id

    @staticmethod
    def _decode_data(data):
        angle = (
            struct.unpack_from(">H", data, 0)[0] * (8 * math.pi / 65535) - 4 * math.pi
        )
        velocity = struct.unpack_from(">H", data, 2)[0] * (60 / 65535) - 30
        torque = struct.unpack_from(">H", data, 4)[0] * (24 / 65535) - 12
        temperature = struct.unpack_from(">H", data, 6)[0] / 10.0
        return angle, velocity, torque, temperature


# Example usage:
if __name__ == "__main__":
    bus = can.interface.Bus(channel="can0", interface="socketcan", bitrate=1_000_000)
    motors = [CyberGearMotor(bus, m) for m in [0x10, 0x7F]]
    for m in motors:
        m.disable()
        print(m.update_state())