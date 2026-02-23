from __future__ import annotations
from dataclasses import dataclass
from typing import List
from pathlib import Path
import xml.etree.ElementTree as ET

@dataclass
class FlowSpec:
    route_id: str
    from_edge: str
    to_edge: str
    begin: float
    end: float
    veh_per_hour: float
    depart_lane: str = "best"
    depart_speed: str = "max"

def write_routes_xml(path: str, flows: List[FlowSpec], vtype_id: str = "car") -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id=vtype_id, accel="2.6", decel="4.5", sigma="0.5", length="5.0", maxSpeed="13.9")
    for f in flows:
        ET.SubElement(
            root, "flow",
            id=f"flow_{f.route_id}_{int(f.begin)}",
            type=vtype_id,
            begin=str(f.begin),
            end=str(f.end),
            from_=f.from_edge,
            to=f.to_edge,
            vehsPerHour=str(f.veh_per_hour),
            departLane=f.depart_lane,
            departSpeed=f.depart_speed,
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
