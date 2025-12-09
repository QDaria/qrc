#!/usr/bin/env python3
"""
Generate Figure 4: Hardware Topology Comparison
Side-by-side visualization of IBM Heavy-Hex vs Rigetti Square Lattice

IBM Heavy-Hex Topology - EXACT reproduction of IBM reference:
- 3 hexagonal unit cells with pointy-top orientation
- Two hexagons on top sharing bottom vertex
- One hexagon on bottom sharing top edges with both top hexagons
- White hollow = degree-2 corner vertices
- Green filled = degree-3 shared vertices
- Blue filled = degree-2 edge qubits (one per edge)
Reference: https://docs.quantum.ibm.com/guides/processor-types
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_figure4():
    """Create architectural diagrams comparing quantum hardware topologies"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========== Panel A: IBM Heavy-Hex Topology ==========
    ax1.set_title('(A) IBM Heron r3 Heavy-Hex Topology\n(156Q fragment showing 3 unit cells)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlim([0, 7])
    ax1.set_ylim([0, 7])
    ax1.axis('off')
    ax1.set_aspect('equal')

    # Center of figure
    cx, cy = 3.5, 3.5

    # Scale
    s = 1.2

    # POINTY-TOP hexagon vertices at angles: 90, 30, 330, 270, 210, 150 degrees
    # (top, top-right, bottom-right, bottom, bottom-left, top-left)
    def hex_vertices_pointy(hx, hy, r):
        """Generate pointy-top hexagon vertices starting from top, going clockwise"""
        angles = [90, 30, -30, -90, -150, 150]  # Pointy top: vertex at top
        return [(hx + r * np.cos(np.radians(a)), hy + r * np.sin(np.radians(a))) for a in angles]

    # IBM's arrangement: inverted Y or triangular with 2 on top, 1 on bottom
    # Hexagon spacing for pointy-top edge-sharing
    hex_h = s * np.sqrt(3)  # horizontal distance between centers
    hex_v = s * 1.5         # vertical distance between centers

    # Three hexagon centers:
    # Top-left hexagon
    h1_cx, h1_cy = cx - hex_h/2, cy + hex_v/2
    # Top-right hexagon
    h2_cx, h2_cy = cx + hex_h/2, cy + hex_v/2
    # Bottom-center hexagon
    h3_cx, h3_cy = cx, cy - hex_v/2

    hex1 = hex_vertices_pointy(h1_cx, h1_cy, s)
    hex2 = hex_vertices_pointy(h2_cx, h2_cy, s)
    hex3 = hex_vertices_pointy(h3_cx, h3_cy, s)

    all_hexagons = [hex1, hex2, hex3]

    # Count vertex occurrences to determine degree
    def round_pt(pt, decimals=2):
        return (round(pt[0], decimals), round(pt[1], decimals))

    vertex_count = {}
    for hex_verts in all_hexagons:
        for v in hex_verts:
            key = round_pt(v)
            vertex_count[key] = vertex_count.get(key, 0) + 1

    # Separate vertices by degree
    corner_verts = []  # degree-2: white/hollow
    shared_verts = []  # degree-3: green

    added_verts = set()
    for hex_verts in all_hexagons:
        for v in hex_verts:
            key = round_pt(v)
            if key not in added_verts:
                added_verts.add(key)
                if vertex_count[key] == 1:
                    corner_verts.append(v)
                else:
                    shared_verts.append(v)

    # Collect unique edges
    edges = []
    added_edges = set()
    for hex_verts in all_hexagons:
        for i in range(6):
            v1 = hex_verts[i]
            v2 = hex_verts[(i+1) % 6]
            edge_key = tuple(sorted([round_pt(v1), round_pt(v2)]))
            if edge_key not in added_edges:
                added_edges.add(edge_key)
                edges.append((v1, v2))

    # Edge qubits at midpoints
    edge_qubits = [((e[0][0]+e[1][0])/2, (e[0][1]+e[1][1])/2) for e in edges]

    # Draw connections first
    for i, (v1, v2) in enumerate(edges):
        eq = edge_qubits[i]
        ax1.plot([v1[0], eq[0]], [v1[1], eq[1]], '-', color='#1f77b4', linewidth=2, alpha=0.8)
        ax1.plot([eq[0], v2[0]], [eq[1], v2[1]], '-', color='#1f77b4', linewidth=2, alpha=0.8)

    # Draw corner vertices (white/hollow - degree 2)
    for (x, y) in corner_verts:
        circle = patches.Circle((x, y), 0.15, fill=False,
                                 edgecolor='#5c5c8a', linewidth=2.5, zorder=10)
        ax1.add_patch(circle)

    # Draw shared vertices (green - degree 3)
    for (x, y) in shared_verts:
        circle = patches.Circle((x, y), 0.15, fill=True, facecolor='#2ca02c',
                                 edgecolor='#1a6b1a', linewidth=2, zorder=10)
        ax1.add_patch(circle)

    # Draw edge qubits (blue - degree 2)
    for (x, y) in edge_qubits:
        circle = patches.Circle((x, y), 0.11, fill=True, facecolor='#1f77b4',
                                 edgecolor='#0d4a7a', linewidth=1.5, zorder=10)
        ax1.add_patch(circle)

    # Dashed extension lines
    for v in corner_verts:
        dx = v[0] - cx
        dy = v[1] - cy
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx, dy = dx/norm * 0.35, dy/norm * 0.35
            ax1.plot([v[0], v[0]+dx], [v[1], v[1]+dy], '--', color='gray', linewidth=1, alpha=0.5)

    # Legend
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='#5c5c8a', label='Corner (deg-2)', linewidth=2),
        patches.Patch(facecolor='#2ca02c', edgecolor='#1a6b1a', label='Shared (deg-3)'),
        patches.Patch(facecolor='#1f77b4', edgecolor='#0d4a7a', label='Edge (deg-2)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Info box - positioned in lower left to avoid overlapping qubits
    textstr = 'Heavy-hex lattice\nMax degree: 3\nReduced crosstalk'
    props = dict(boxstyle='round', facecolor='#deebf7', alpha=0.8)
    ax1.text(1.0, 0.5, textstr, ha='left', fontsize=10, bbox=props)

    # ========== Panel B: Rigetti Square Lattice ==========
    ax2.set_title('(B) Rigetti Novera Square Lattice\n(9Q full nearest-neighbor)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim([-1, 4])
    ax2.set_ylim([-1, 4])
    ax2.axis('off')

    # 3×3 grid layout
    for i in range(3):
        for j in range(3):
            x, y = j * 1.2 + 0.5, (2 - i) * 1.2 + 0.5

            # Draw qubit
            circle = patches.Circle((x, y), 0.3, fill=True, facecolor='lightgreen',
                                     edgecolor='black', linewidth=2)
            ax2.add_patch(circle)
            ax2.text(x, y, f'Q{i*3+j}', ha='center', va='center',
                     fontsize=9, fontweight='bold')

            # Draw connections to right neighbor
            if j < 2:
                ax2.plot([x + 0.3, x + 0.9], [y, y], 'k-', linewidth=2, alpha=0.7)

            # Draw connections to bottom neighbor
            if i < 2:
                ax2.plot([x, x], [y - 0.3, y - 0.9], 'k-', linewidth=2, alpha=0.7)

    # Add explanatory text box
    textstr = 'Full nearest-neighbor\nNo SWAP gates needed\nDense coupling map'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
    ax2.text(1.7, -0.7, textstr, ha='center', fontsize=11, bbox=props)

    # Add connectivity metric
    total_edges_rigetti = 12
    max_edges_rigetti = 9 * 8 / 2
    connectivity_rigetti = total_edges_rigetti / max_edges_rigetti * 100
    ax2.text(1.7, 3.5, f'Connectivity: {connectivity_rigetti:.1f}%',
             ha='center', fontsize=10, style='italic')

    # Add overall comparison note
    fig.text(0.5, 0.02,
             'Key Insight: Square lattice topology reduces gate overhead and improves sample efficiency',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save in multiple formats
    plt.savefig('figures/figure4_topology_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure4_topology_comparison.pdf', bbox_inches='tight')

    print("Figure 4 generated successfully!")
    print("  - PNG: figures/figure4_topology_comparison.png")
    print("  - PDF: figures/figure4_topology_comparison.pdf")

    plt.close()

if __name__ == '__main__':
    generate_figure4()
