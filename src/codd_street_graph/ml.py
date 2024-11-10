import uuid
import networkx as nx
import math
import numpy as np
import geopandas
import shapely

R = 5
L_MIN = 100
L_MAX = 10000
K = 0.75
MAX_FLOW = 800


def angle_between(u, v):
    dot_product = sum(i * j for i, j in zip(u, v))
    norm_u = math.sqrt(sum(i**2 for i in u))
    norm_v = math.sqrt(sum(i**2 for i in v))
    cos_theta = dot_product / (norm_u * norm_v)
    angle_rad = math.acos(cos_theta)
    return np.cos(angle_rad) ** 50


def union_nodes(G, nodes, centroid, idx_to_id_dfi):
    edges = [e + (G.edges[e],) for e in G.edges if e[0] in nodes or e[1] in nodes]
    uid = uuid.uuid4()
    idx_to_id_dfi.update({G.nodes[i]["index"]: uid for i in nodes})

    G.remove_nodes_from(nodes)
    G.add_node(
        uid, index=-1, type="street", coords=np.array(centroid.coords)[0], flow=0.0
    )

    for e in edges:
        e = list(e)
        if e[0] in nodes:
            e[0] = uid
        if e[1] in nodes:
            e[1] = uid
        if e[1] != e[0]:
            assert e[1] in G.nodes and e[0] in G.nodes
            G.add_edge(e[0], e[1], **e[2])


def distance_along_line(line, p1, p2):
    pos1 = line.project(p1)
    pos2 = line.project(p2)
    return abs(pos1 - pos2)


def get_potoc(df, G, home_index):  # [midages, children schul ages]
    # indes_for_df=G.nodes[home_index]["index"]
    if df["Type"][home_index] == "Жилые дома":
        count_man = df["Apartments"][home_index] * 3 * 45 / 100
        return [count_man * 45 / 100, count_man * 5 / 100]
    else:
        return [0, 0]


def cost_function(G, a, b, sig=2):
    # чем больше значение sig, тем разнообразнее пути (оптимально [0.5; 2])
    def f(u, v, data):
        return np.exp(
            np.random.normal(
                scale=sig**2,
                loc=np.log(
                    data["weight"]
                    * angle_between(
                        (G.nodes[b]["coords"] - G.nodes[a]["coords"]),
                        (G.nodes[v]["coords"] - G.nodes[u]["coords"]),
                    )
                ),
            )
        )

    return f


def find_shortest_nondet(G, a, b):
    return nx.shortest_path(
        G, a, b, weight=cost_function(G, a, b), method="bellman-ford"
    )


def get_all_varies(a, b, G):
    try:
        p1 = nx.shortest_path(G, a, b, weight=lambda u, v, data: data["weight"])
        p2 = find_shortest_nondet(G, a, b)
        return [p1, p2]
    except Exception:
        return [[a], [a]]


def house_crossing(a, b, houses):
    line = shapely.LineString([a, b])
    for house in houses:
        if shapely.intersects(line, house):
            return True
    return False


def road_crossing(a, b, roads):
    line = shapely.LineString([a, b])
    for road in roads:
        if shapely.intersects(line, road):
            return True
    return False


def is_valid_walkway(walkway, roads, houses):
    if walkway[2] < L_MIN or walkway[2] > L_MAX:
        return False

    if walkway[0].distance(walkway[1]) > K * walkway[2]:
        return False

    if road_crossing(walkway[0], walkway[1], roads):
        return False

    if house_crossing(walkway[0], walkway[1], houses):
        return False

    return True


def start(df, G, idx_to_id_df1, idx_to_id_df3):
    for ind in df[df.Type == "Жилые дома"].index:  # for midage
        midage, other = get_potoc(df, G, ind)

        start_node = idx_to_id_df1[ind]
        end_node = idx_to_id_df3[df.loc[ind, "nearest_df3"]]

        p1, p2 = get_all_varies(start_node, end_node, G)
        edge_path1 = [[p1[i - 1], p1[i]] for i in range(1, len(p1))]
        edge_path2 = [[p2[i - 1], p2[i]] for i in range(1, len(p2))]
        for edge in edge_path1:
            u = edge[0]
            v = edge[1]
            # assert False, (u, v)
            G.edges[u, v]["flow"] += midage * 30 / 100
        for edge in edge_path2:
            u = edge[0]
            v = edge[1]
            G.edges[u, v]["flow"] += midage * 60 / 100

        end_node = idx_to_id_df1[df.loc[ind, "nearest_school"]]
        p1, p2 = get_all_varies(start_node, end_node, G)
        edge_path1 = [[p1[i - 1], p1[i]] for i in range(1, len(p1))]
        edge_path2 = [[p2[i - 1], p2[i]] for i in range(1, len(p2))]
        for edge in edge_path2:
            u = edge[0]
            v = edge[1]
            G.edges[u, v]["flow"] += midage

        end_node = idx_to_id_df1[df.loc[ind, "nearest_preschool"]]
        p1, p2 = get_all_varies(start_node, end_node, G)
        edge_path1 = [[p1[i - 1], p1[i]] for i in range(1, len(p1))]
        edge_path2 = [[p2[i - 1], p2[i]] for i in range(1, len(p2))]
        for edge in edge_path2:
            u = edge[0]
            v = edge[1]
            G.edges[u, v]["flow"] += midage

        end_node = idx_to_id_df1[df.loc[ind, "nearest_administrative"]]
        p1, p2 = get_all_varies(start_node, end_node, G)
        edge_path1 = [[p1[i - 1], p1[i]] for i in range(1, len(p1))]
        edge_path2 = [[p2[i - 1], p2[i]] for i in range(1, len(p2))]
        for edge in edge_path1:
            u = edge[0]
            v = edge[1]
            G.edges[u, v]["flow"] += midage
    return G


def process(file_name1, file_name2, file_name3):
    df1 = geopandas.read_file(file_name1)
    df2 = geopandas.read_file(file_name2)
    df3 = geopandas.read_file(file_name3)

    minx, miny, maxx, maxy = df1.total_bounds

    df2 = df2.cx[minx:maxx, miny:maxy]
    df3 = df3.cx[minx:maxx, miny:maxy]

    df2.fillna(0, inplace=True)
    df2.reset_index(inplace=True, drop=True)

    ps = []
    for i in df2.index:
        for j in df2.index:
            if i < j:
                ps.append(
                    (df2.loc[i, "geometry"].intersection(df2.loc[j, "geometry"]), i, j)
                )

    data = [_ for geom, *_ in ps if geom.geom_type == "Point"]
    ps = [geom for geom, *_ in ps if geom.geom_type == "Point"]

    dfi = geopandas.GeoDataFrame(data=data, geometry=ps, crs=df2.crs)

    graph_size = dfi.shape[0] + df3.shape[0] + df1.shape[0]
    ids = [str(uuid.uuid4()) for _ in range(graph_size)]

    gr = [[] for _ in range(dfi.shape[0])]

    for i in df2.index:
        psi = [j for j, (a1, a2, _) in dfi.iterrows() if i in (a1, a2)]
        for j in range(len(psi)):
            for k in range(j + 1, len(psi)):
                d = distance_along_line(
                    df2.loc[i, "geometry"],
                    dfi.loc[psi[j], "geometry"],
                    dfi.loc[psi[k], "geometry"],
                )
                gr[psi[j]].append((psi[k], d))
                gr[psi[k]].append((psi[j], d))

    idx_to_id_dfi = {i: ids.pop() for i in dfi.index}

    G = nx.Graph()

    G.add_nodes_from(
        [
            (
                idx_to_id_dfi[i],
                {
                    "type": "street",
                    "index": i,
                    "coords": np.array(dfi.loc[i, "geometry"].coords)[0],
                },
            )
            for i in dfi.index
        ]
    )

    for i in dfi.index:
        G.add_weighted_edges_from(
            [(idx_to_id_dfi[i], idx_to_id_dfi[j], d) for j, d in gr[i]]
        )

    df3["nearest"] = df3.geometry.apply(
        lambda g: dfi.sindex.nearest(g, return_all=False)[1, 0]
    )

    idx_to_id_df3 = {i: ids.pop() for i in df3.index}

    G.add_nodes_from(
        [
            (
                idx_to_id_df3[i],
                {
                    "type": "stop",
                    "index": i,
                    "coords": np.array(df3.loc[i, "geometry"].coords)[0],
                },
            )
            for i in df3.index
        ]
    )

    for i, s in df3.iterrows():
        G.add_weighted_edges_from(
            [
                (
                    idx_to_id_df3[i],
                    idx_to_id_dfi[s["nearest"]],
                    dfi.loc[s["nearest"], "geometry"].distance(s["geometry"]),
                )
            ]
        )

    df1["nearest"] = df1.geometry.centroid.apply(
        lambda g: dfi.sindex.nearest(g, return_all=False)[1, 0]
    )

    idx_to_id_df1 = {i: ids.pop() for i in df1.index}

    G.add_nodes_from(
        [
            (
                idx_to_id_df1[i],
                {
                    "type": "building",
                    "index": i,
                    "coords": np.array(df1.loc[i, "geometry"].centroid.coords)[0],
                },
            )
            for i in df1.index
        ]
    )

    for i, s in df1.iterrows():
        G.add_weighted_edges_from(
            [
                (
                    idx_to_id_df1[i],
                    idx_to_id_dfi[s["nearest"]],
                    dfi.loc[s["nearest"], "geometry"].distance(s["geometry"]),
                )
            ]
        )

    circ = geopandas.GeoDataFrame(geometry=dfi.geometry.buffer(R), crs=dfi.crs)
    not_covered = np.array([True] * dfi.shape[0])
    circ["point_numbers"] = circ.geometry.apply(
        lambda g: dfi[dfi.geometry.covered_by(g)].shape[0]
    )

    while any(not_covered):
        i = circ.point_numbers.idxmax()

        circ.loc[i, "point_numbers"] = -1
        circ.loc[i, "points"] = ",".join(
            map(
                str,
                (
                    dfi[
                        (dfi.geometry.covered_by(circ.loc[i, "geometry"])) & not_covered
                    ].index
                ),
            )
        )
        not_covered = not_covered & (
            ~dfi.loc[not_covered, "geometry"].covered_by(circ.loc[i, "geometry"])
        )

    circ = circ[circ.points != ""]

    for g, _, points in circ.itertuples(index=False):
        union_nodes(
            G,
            [idx_to_id_dfi[i] for i in map(int, points.split(","))],
            g.centroid,
            idx_to_id_dfi,
        )

    for e in G.edges:
        G.edges[e]["flow"] = 0

    df1 = df1.sjoin_nearest(df3[["geometry"]], how="left").rename(
        columns={"index_right": "nearest_df3"}
    )
    df1 = df1.sjoin_nearest(
        df1.loc[df1.Type == "Школы", ["geometry"]], how="left"
    ).rename(columns={"index_right": "nearest_school"})
    df1 = df1.sjoin_nearest(
        df1.loc[df1.Type == "Дошкольные", ["geometry"]], how="left"
    ).rename(columns={"index_right": "nearest_preschool"})
    df1 = df1.sjoin_nearest(
        df1.loc[df1.Type == "Административные сооружения", ["geometry"]], how="left"
    ).rename(columns={"index_right": "nearest_administrative"})

    G = start(df1, G, idx_to_id_df1, idx_to_id_df3)

    G_OLD = G.copy()

    roads = df2[df2.Car == 1].cx[minx:maxx, miny:maxy].geometry
    houses = df1.geometry

    for i in G.nodes:
        G.nodes[i]["coords"] = shapely.Point(G.nodes[i]["coords"])

    df2.loc[:, "updated"] = 0

    for edge in G.edges.data("flow"):
        if edge[2] > MAX_FLOW:
            interesting_edges = G.edges([edge[0], edge[1]], data="flow")

            interesting_edges_left = G.edges(edge[0], data="flow")
            interesting_edges_right = G.edges(edge[1], data="flow")

            new_walkways = []

            for good_edge in interesting_edges_left:
                if good_edge[1] == edge[1]:
                    continue
                old_dist = (
                    G.get_edge_data(edge[0], edge[1])["weight"]
                    + G.get_edge_data(good_edge[0], good_edge[1])["weight"]
                )
                new_walkways.append((edge[1], good_edge[1], good_edge[2], old_dist))

            for good_edge in interesting_edges_right:
                if good_edge[1] == edge[0]:
                    continue
                old_dist = (
                    G.get_edge_data(edge[0], edge[1])["weight"]
                    + G.get_edge_data(good_edge[0], good_edge[1])["weight"]
                )
                new_walkways.append((edge[0], good_edge[1], good_edge[2], old_dist))

            # print(new_walkways)

            new_walkways.sort(key=lambda x: -x[2])

            # print(new_walkways)

            for i in range(len(new_walkways)):
                walkway = new_walkways[i]
                walkway = (
                    G.nodes[walkway[0]]["coords"],
                    G.nodes[walkway[1]]["coords"],
                    walkway[3],
                )

                if is_valid_walkway(walkway, roads, houses):
                    print(walkway)
                    df2.loc[len(df2.index)] = [
                        None,
                        None,
                        None,
                        "Пешеходные дорожки",
                        "Any",
                        None,
                        None,
                        None,
                        4,
                        4,
                        4,
                        4,
                        1,
                        0,
                        shapely.geometry.LineString([walkway[0], walkway[1]]),
                        1,
                    ]
                    break

    ps = []
    for i in df2.index:
        for j in df2.index:
            if i < j:
                ps.append(
                    (df2.loc[i, "geometry"].intersection(df2.loc[j, "geometry"]), i, j)
                )

    data = [_ for geom, *_ in ps if geom.geom_type == "Point"]
    ps = [geom for geom, *_ in ps if geom.geom_type == "Point"]

    dfi = geopandas.GeoDataFrame(data=data, geometry=ps, crs=df2.crs)

    graph_size = dfi.shape[0] + df3.shape[0] + df1.shape[0]
    ids = [str(uuid.uuid4()) for _ in range(graph_size)]

    gr = [[] for _ in range(dfi.shape[0])]

    for i, dt in df2.iterrows():
        psi = [j for j, (a1, a2, _) in dfi.iterrows() if i in (a1, a2)]
        for j in range(len(psi)):
            for k in range(j + 1, len(psi)):
                d = distance_along_line(
                    df2.loc[i, "geometry"],
                    dfi.loc[psi[j], "geometry"],
                    dfi.loc[psi[k], "geometry"],
                )
                gr[psi[j]].append((psi[k], d, dt["updated"], dt["Car"]))
                gr[psi[k]].append((psi[j], d, dt["updated"], dt["Car"]))

    idx_to_id_dfi = {i: ids.pop() for i in dfi.index}

    G = nx.Graph()

    G.add_nodes_from(
        [
            (
                idx_to_id_dfi[i],
                {
                    "type": "street",
                    "index": i,
                    "coords": np.array(dfi.loc[i, "geometry"].coords)[0],
                },
            )
            for i in dfi.index
        ]
    )

    for i in dfi.index:
        for j, d, upd, cr in gr[i]:
            G.add_edge(
                idx_to_id_dfi[i], idx_to_id_dfi[j], weight=d, update=upd, is_car=cr
            )

    df3["nearest"] = df3.geometry.apply(
        lambda g: dfi.sindex.nearest(g, return_all=False)[1, 0]
    )

    idx_to_id_df3 = {i: ids.pop() for i in df3.index}

    G.add_nodes_from(
        [
            (
                idx_to_id_df3[i],
                {
                    "type": "stop",
                    "index": i,
                    "coords": np.array(df3.loc[i, "geometry"].coords)[0],
                },
            )
            for i in df3.index
        ]
    )

    for i, s in df3.iterrows():
        G.add_weighted_edges_from(
            [
                (
                    idx_to_id_df3[i],
                    idx_to_id_dfi[s["nearest"]],
                    dfi.loc[s["nearest"], "geometry"].distance(s["geometry"]),
                )
            ]
        )

    df1["nearest"] = df1.geometry.centroid.apply(
        lambda g: dfi.sindex.nearest(g, return_all=False)[1, 0]
    )

    idx_to_id_df1 = {i: ids.pop() for i in df1.index}

    G.add_nodes_from(
        [
            (
                idx_to_id_df1[i],
                {
                    "type": "building",
                    "index": i,
                    "coords": np.array(df1.loc[i, "geometry"].centroid.coords)[0],
                },
            )
            for i in df1.index
        ]
    )

    for i, s in df1.iterrows():
        G.add_weighted_edges_from(
            [
                (
                    idx_to_id_df1[i],
                    idx_to_id_dfi[s["nearest"]],
                    dfi.loc[s["nearest"], "geometry"].distance(s["geometry"]),
                )
            ]
        )

    circ = geopandas.GeoDataFrame(geometry=dfi.geometry.buffer(R), crs=dfi.crs)
    not_covered = np.array([True] * dfi.shape[0])
    circ["point_numbers"] = circ.geometry.apply(
        lambda g: dfi[dfi.geometry.covered_by(g)].shape[0]
    )

    while any(not_covered):
        i = circ.point_numbers.idxmax()

        circ.loc[i, "point_numbers"] = -1
        circ.loc[i, "points"] = ",".join(
            map(
                str,
                (
                    dfi[
                        (dfi.geometry.covered_by(circ.loc[i, "geometry"])) & not_covered
                    ].index
                ),
            )
        )
        not_covered = not_covered & (
            ~dfi.loc[not_covered, "geometry"].covered_by(circ.loc[i, "geometry"])
        )

    circ = circ[circ.points != ""]

    for g, _, points in circ.itertuples(index=False):
        union_nodes(
            G,
            [idx_to_id_dfi[i] for i in map(int, points.split(","))],
            g.centroid,
            idx_to_id_dfi,
        )

    for e in G.edges:
        G.edges[e]["flow"] = 0

    G = start(df1, G, idx_to_id_df1, idx_to_id_df3)

    for e in G.edges:
        if G.edges[e].get("is_car", 1) == 0 and G.edges[e]["flow"] <= 50:
            G.edges[e]["update"] = -1

    return G_OLD, G, df1


def get_metrix(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.0]
    e_max_large = [
        (u, v) for (u, v, d) in G.edges(data=True) if d["flow"] / d["weight"] > 0.0
    ]
    return (len(elarge), len(e_max_large))
