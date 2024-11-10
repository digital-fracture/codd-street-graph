from functools import cached_property

from pydantic import BaseModel, PositiveInt, computed_field

from config import default_position, default_max_flow

Position = tuple[float, float]
Polygon = list[Position]


class StreetGraphEdge(BaseModel):
    id: PositiveInt
    flow: PositiveInt
    max_flow: PositiveInt = default_max_flow
    load_percentage: float
    year: PositiveInt = 0  # initialized from StreetGraph


class StreetGraphNode(BaseModel):
    id: PositiveInt
    name: str
    position: Position
    connections: list[StreetGraphEdge]


class StreetGraph(BaseModel):
    nodes: list[StreetGraphNode]
    buildings: list[Polygon]
    year: PositiveInt
    hot_points: list[Position]

    @computed_field
    @cached_property
    def position(self) -> Position:
        if not self.nodes:
            return default_position

        latitude_sum = sum(node.position[0] for node in self.nodes)
        longitude_sum = sum(node.position[1] for node in self.nodes)

        return latitude_sum / len(self.nodes), longitude_sum / len(self.nodes)

    def model_post_init(self, __context: ...) -> None:
        for node in self.nodes:
            for edge in node.connections:
                edge.year = self.year


class StreetGraphPair(StreetGraph):
    update_nodes: list[StreetGraphNode]

    def model_post_init(self, __context: ...) -> None:
        super().model_post_init(__context)

        for node in self.update_nodes:
            for edge in node.connections:
                edge.year = self.year


class GraphHistory(BaseModel):
    graphs: list[StreetGraphPair]

    @computed_field
    @cached_property
    def position(self) -> Position:
        return self.graphs[-1].position if self.graphs else default_position


class PopulationDistribution(BaseModel):
    children_percentage: float
    retiree_percentage: float
    adult_personal_transport_percentage: float
    adult_public_transport_percentage: float
    adult_other_percentage: float
