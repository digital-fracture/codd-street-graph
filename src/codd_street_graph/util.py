import networkx as nx
import geopandas

from .models import StreetGraphNode, StreetGraphEdge, StreetGraphPair


def serialize(old_graph: nx.Graph, new_graph: nx.Graph, houses: geopandas.GeoDataFrame) -> StreetGraphPair:
    old_nodes = []
    for id, nx_node in enumerate(old_graph.nodes, start=1):
        old_nodes.append(
            StreetGraphNode(
                id=id,
                name=str(id),
                position=old_graph.nodes[nx_node]["coords"],
                connections=[],
            )
        )

    for nx_edge in old_graph.edges:
        start_node_index = nx_edge[0]
        end_node_id = nx_edge[1] + 1

        flow = old_graph.edges[nx_edge]["flow"]
        max_flow = 800
        load_percentage = flow / max_flow

        old_nodes[start_node_index].connections.append(
            StreetGraphEdge(
                id=end_node_id,
                flow=flow,
                max_flow=max_flow,
                load_percentage=load_percentage,
            )
        )

    new_nodes = []
    for id, nx_node in enumerate(new_graph.nodes, start=1):
        new_nodes.append(
            StreetGraphNode(
                id=id,
                name=str(id),
                position=new_graph.nodes[nx_node]["coords"],
                connections=[],
            )
        )

    for nx_edge in new_graph.edges:
        start_node_index = nx_edge[0]
        end_node_id = nx_edge[1] + 1

        flow = new_graph.edges[nx_edge]["flow"]
        max_flow = 800
        load_percentage = flow / max_flow

        new_nodes[start_node_index].connections.append(
            StreetGraphEdge(
                id=end_node_id,
                flow=flow,
                max_flow=max_flow,
                load_percentage=load_percentage,
            )
        )

    buildings = []
    for house in houses:
        buildings.append(house.geometry.points)

    return StreetGraphPair(
        nodes=old_nodes,
        update_nodes=new_nodes,
        buildings=buildings,
        hot_points=[],
    )
