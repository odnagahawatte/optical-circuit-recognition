import re
import itertools


def evaluate_netlist(netlist_strings):
    gates = {}

    # 1. PARSE THE TEXT NETLIST
    pattern = r"([A-Z]+) Gate \(ID:(\d+)\) has an (INPUT|OUTPUT) connected to Wire #(\d+)"

    for line in netlist_strings:
        match = re.search(pattern, line)
        if match:
            g_type, g_id, pin_type, wire_id = match.groups()
            g_id, wire_id = int(g_id), int(wire_id)

            if g_id not in gates:
                gates[g_id] = {'type': g_type, 'inputs': [], 'outputs': []}

            if pin_type == "INPUT":
                gates[g_id]['inputs'].append(wire_id)
            else:
                gates[g_id]['outputs'].append(wire_id)

    # --- NEW: PRE-FLIGHT TOPOLOGY CHECK ---
    print("\n" + "=" * 45)
    print("           COMPILER DIAGNOSTICS")
    print("=" * 45)
    topology_safe = True
    for g_id, g in gates.items():
        if len(g['inputs']) == 0:
            print(f"[-] ERROR: {g['type']} Gate (ID:{g_id}) has NO inputs connected.")
            topology_safe = False
        if g['type'] == 'NOT' and len(g['inputs']) > 1:
            print(f"[-] ERROR: NOT Gate (ID:{g_id}) has too many inputs (Max 1).")
            topology_safe = False

    if not topology_safe:
        print("\n[!] FATAL: Circuit topology is physically broken. Halting simulation.")
        print("[!] Advice: Ensure drawing uses solid marker strokes and touches the gates.")
        return
    print("[+] Topology verified. All gates have valid input constraints.")
    # --------------------------------------

    # 2. IDENTIFY GLOBAL INPUTS & OUTPUTS
    all_inputs = set(w for g in gates.values() for w in g['inputs'])
    all_outputs = set(w for g in gates.values() for w in g['outputs'])

    global_inputs = sorted(list(all_inputs - all_outputs))
    global_outputs = sorted(list(all_outputs - all_inputs))

    if not global_inputs:
        print("[-] Error: No global inputs detected. Circuit might be a closed loop.")
        return

    print("\n" + "=" * 45)
    print("           LOGIC SIMULATION (TRUTH TABLE)")
    print("=" * 45)

    header_in = [f"W{w}" for w in global_inputs]
    header_out = [f"W{w}" for w in global_outputs]
    print(f"{' | '.join(header_in)} || {' | '.join(header_out)}")
    print("-" * 45)

    # 3. GENERATE BINARY COMBINATIONS & SIMULATE
    for combo in itertools.product([0, 1], repeat=len(global_inputs)):
        wire_states = {w: val for w, val in zip(global_inputs, combo)}
        evaluated_gates = set()

        while len(evaluated_gates) < len(gates):
            progress = False
            for g_id, g in gates.items():
                if g_id in evaluated_gates:
                    continue

                # The length check (len > 0) is now guaranteed by the Pre-Flight check,
                # but we keep the logic intact here for safety.
                if all(w in wire_states for w in g['inputs']):
                    in_vals = [wire_states[w] for w in g['inputs']]

                    if g['type'] == 'AND':
                        res = 1 if all(v == 1 for v in in_vals) else 0
                    elif g['type'] == 'OR':
                        res = 1 if any(v == 1 for v in in_vals) else 0
                    elif g['type'] == 'NOT':
                        res = 1 if in_vals[0] == 0 else 0
                    else:
                        res = 0

                    for out_w in g['outputs']:
                        wire_states[out_w] = res

                    evaluated_gates.add(g_id)
                    progress = True

            if not progress:
                # Silently break for this row if simulation stalls due to missing intermediate wires
                break

        # 4. PRINT THE ROW (Only if the simulation successfully completed the DAG)
        if len(evaluated_gates) == len(gates):
            row_in = [str(wire_states[w]) for w in global_inputs]
            row_out = [str(wire_states.get(w, 'X')) for w in global_outputs]

            in_str = ' | '.join(f"{val:^{len(header_in[i])}}" for i, val in enumerate(row_in))
            out_str = ' | '.join(f"{val:^{len(header_out[i])}}" for i, val in enumerate(row_out))
            print(f"{in_str} || {out_str}")