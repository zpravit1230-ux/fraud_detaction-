"""Interactive PyVis rendering for vendor relationship neighborhoods."""

from __future__ import annotations

import tempfile

import streamlit as st
from pyvis.network import Network

from src.graph_engine import vendor_neighborhood


def render_graph(graph, vendor_id: str, depth: int = 2, entity_types: set[str] | None = None, height: int = 560) -> None:
    subgraph = vendor_neighborhood(graph, vendor_id, depth, entity_types)
    if subgraph.number_of_nodes() == 0:
        st.info("No connected entities are available for this selection.")
        return
    network = Network(height=f"{height}px", width="100%", bgcolor="#101b20", font_color="#edf2ef", directed=False, notebook=False, cdn_resources="in_line")
    network.barnes_hut(gravity=-2800, central_gravity=0.28, spring_length=130, spring_strength=0.035, damping=0.12)
    for node, attrs in subgraph.nodes(data=True):
        kind = attrs.get("entity_type", "Entity")
        network.add_node(str(node), label=str(attrs.get("label", node)), title=str(attrs.get("title", f"{kind}: {node}")),
                         color=attrs.get("color", "#82909a"), size=31 if kind == "Vendor" and node == vendor_id else 19 if kind == "Vendor" else 13,
                         shape="dot", group=kind)
    for left, right, attrs in subgraph.edges(data=True):
        network.add_edge(str(left), str(right), title=attrs.get("relationship", "RELATED_TO"), color="#42555c", width=1.4)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", encoding="utf-8", delete=False) as output:
        output.write(network.generate_html(name=output.name, local=False, notebook=False))
        output.flush()
        html_path = output.name
    with open(html_path, encoding="utf-8") as graph_file:
        st.components.v1.html(graph_file.read(), height=height, scrolling=False)