"""
sdg_pipeline/db_bahn/tau2_domain/data_model.py
==============================================
Pydantic world-state for the tau2 `db_bahn` domain. Field names match exactly the JSON produced by
`sdg_pipeline/db_bahn/seed_worldstate.py` (BahnDB is a tau2 `DB` = BaseModelNoExtra → NO extra keys).
Entity tables are dicts keyed by pk; relational tables are lists.

Requires the tau2 (Python 3.12) venv — only imported in the tau2 context.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from tau2.environment.db import DB


class Station(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    eva: Optional[str] = None


class Line(BaseModel):
    line_id: str
    product: str
    short_name: str


class Trip(BaseModel):
    trip_id: str
    line_id: str
    service_id: str
    zugnummer: str
    product: str
    origin_station: str
    dest_station: str
    headsign: str
    dep_time: str
    arr_time: str
    vehicle_id: str


class ScheduleStop(BaseModel):
    trip_id: str
    seq: int
    station_id: str
    arr: str
    dep: str
    headsign: str


class Delay(BaseModel):
    trip_id: str
    seq: int
    delay_sec: int
    cause: str
    remark: str


class Position(BaseModel):
    trip_id: str
    lat: float
    lon: float
    progress_pct: float
    next_station_id: Optional[str] = None
    at_station: Optional[str] = None
    source: str


class Vehicle(BaseModel):
    vehicle_id: str
    type: str
    capacity: int
    home_depot: str


class MaintenanceOrder(BaseModel):
    order_id: str
    vehicle_id: str
    type: str
    status: str
    depot: str
    due_at: str
    severity: str


class Employee(BaseModel):
    emp_id: str
    name: str
    role: str
    home_base: str
    qualifications: List[str]


class Shift(BaseModel):
    shift_id: str
    emp_id: str
    start: str
    end: str
    base: str


class Assignment(BaseModel):
    assignment_id: str
    trip_id: str
    emp_id: str
    role: str


class BahnDB(DB):
    """Frozen Deutsche-Bahn world-state (real gtfs.de de_fv seed + synthetic seeded tables)."""

    # Wave 3: pending transient-fault counters {tool_name: remaining_failures}, set via
    # inject_transient_stoerung. PrivateAttr on purpose: NOT part of model_dump(), so the
    # verifier's db_match/no_write hash comparison stays blind to consumed counters and
    # db.json serialization is unchanged. model_copy(deep=True) carries it (pydantic v2).
    _transient: Dict[str, int] = PrivateAttr(default_factory=dict)

    meta: dict = Field(default_factory=dict)
    stations: Dict[str, Station] = Field(default_factory=dict)
    lines: Dict[str, Line] = Field(default_factory=dict)
    trips: Dict[str, Trip] = Field(default_factory=dict)
    vehicles: Dict[str, Vehicle] = Field(default_factory=dict)
    maintenance_orders: Dict[str, MaintenanceOrder] = Field(default_factory=dict)
    employees: Dict[str, Employee] = Field(default_factory=dict)
    schedule: List[ScheduleStop] = Field(default_factory=list)
    delays: List[Delay] = Field(default_factory=list)
    positions: List[Position] = Field(default_factory=list)
    shifts: List[Shift] = Field(default_factory=list)
    assignments: List[Assignment] = Field(default_factory=list)
