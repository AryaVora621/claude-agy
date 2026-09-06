"""
ASCII Terminal Dashboard and Cluster Visualizer for SwarmRaft.
Renders real-time node states, election terms, logs, state machine store,
and network partition topologies.
"""

from __future__ import annotations
from typing import Dict, List, Any
from swarmraft.types import NodeRole
from swarmraft.cluster import RaftCluster


def render_cluster_ascii(cluster: RaftCluster) -> str:
    """Renders an ASCII dashboard snapshot of the Raft cluster."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                   SWARMRAFT: DISTRIBUTED CONSENSUS DASHBOARD                 ║")
    lines.append("║         Ongaro-Ousterhout Raft Protocol & Replicated State Machine           ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")

    # 1. Cluster Status Table
    lines.append("\n[1] Cluster Nodes & Consensus Roles:")
    lines.append("┌──────────┬────────────┬──────┬──────────┬────────┬─────────┬─────────┬─────────┐")
    lines.append("│ Node ID  │ Status     │ Term │ Role     │ Voted  │ Commit  │ Applied │ Log Len │")
    lines.append("├──────────┼────────────┼──────┼──────────┼────────┼─────────┼─────────┼─────────┤")

    leader_id = None
    for nid in cluster.node_ids:
        node = cluster.nodes.get(nid)
        if not node:
            continue

        is_alive = cluster.transport.is_node_alive(nid)
        status_str = "ONLINE" if is_alive else "CRASHED"

        role_str = node.role.value if is_alive else "OFFLINE"
        if node.role == NodeRole.LEADER and is_alive:
            role_str = "★ LEADER"
            leader_id = nid

        term = node.current_term
        voted = node.voted_for or "-"
        commit_idx = node.commit_index
        applied_idx = node.last_applied
        log_len = len(node.log) + node.last_included_index

        lines.append(
            f"│ {nid:<8} │ {status_str:<10} │ {term:>4} │ {role_str:<8} │ {voted:<6} │ "
            f"{commit_idx:>7} │ {applied_idx:>7} │ {log_len:>7} │"
        )

    lines.append("└──────────┴────────────┴──────┴──────────┴────────┴─────────┴─────────┴─────────┘")

    # 2. Partition Topology
    lines.append("\n[2] Network Partition Topology:")
    groups: Dict[int, List[str]] = {}
    for nid in cluster.node_ids:
        part_idx = cluster.transport._node_partition_map.get(nid, 0)
        groups.setdefault(part_idx, []).append(nid)

    if len(groups) <= 1:
        lines.append("  ● Full Mesh Connectivity (No active network partitions)")
    else:
        part_strs = []
        for g_idx, members in sorted(groups.items()):
            part_strs.append(f"Partition #{g_idx}: [{', '.join(members)}]")
        lines.append("  ⚡ " + "  <== SPLIT BRAIN ==>  ".join(part_strs))

    # 3. Log Timeline View
    lines.append("\n[3] Replicated Log Timeline (Index:Term:Command):")
    for nid in cluster.node_ids:
        node = cluster.nodes.get(nid)
        if not node:
            continue

        entries_str = []
        if node.last_included_index > 0:
            entries_str.append(f"[Snapshot<=Idx#{node.last_included_index}]")

        for e in node.log[-8:]:  # Show up to last 8 entries
            is_committed = e.index <= node.commit_index
            tag = "✓" if is_committed else "○"
            cmd_summary = str(e.command)
            if isinstance(e.command, dict):
                op = e.command.get("op", "")
                k = e.command.get("key", "")
                v = e.command.get("val", "")
                cmd_summary = f"{op} {k}={v}" if v else f"{op} {k}"
            entries_str.append(f"[{tag} #{e.index}:T{e.term}|{cmd_summary}]")

        log_display = " ".join(entries_str) if entries_str else "(empty log)"
        lines.append(f"  {nid:<8}: {log_display}")

    # 4. State Machine Store State (from Leader or quorum)
    lines.append("\n[4] Replicated Key-Value State Machine:")
    ref_node = cluster.get_leader() or (list(cluster.nodes.values())[0] if cluster.nodes else None)
    if ref_node and hasattr(ref_node.state_machine, "to_dict"):
        kv_data = ref_node.state_machine.to_dict()
        if kv_data:
            kv_items = [f"'{k}': {repr(v)}" for k, v in sorted(kv_data.items())]
            lines.append("  Data: { " + ", ".join(kv_items) + " }")
        else:
            lines.append("  Data: { } (Empty store)")
    else:
        lines.append("  Data: (No reachable node)")

    # 5. Network Telemetry
    lines.append("\n[5] Network RPC Telemetry:")
    stats = cluster.transport.stats
    lines.append(
        f"  Total RPCs Sent: {stats.total_rpcs_sent:,}  │  "
        f"Delivered: {stats.total_rpcs_delivered:,}  │  "
        f"Dropped (Partitions/Faults): {stats.total_rpcs_dropped:,}"
    )

    return "\n".join(lines)
