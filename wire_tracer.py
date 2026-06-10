import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_netlist(gates, routing_matrix, original_img):
    # 1. Setup hitboxes and IDs for the gates
    for i, g in enumerate(gates):
        g["id"] = i
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]

        # 15-pixel "Sensor Ring"
        pad = 15
        g["hitbox"] = (max(0, x - pad), max(0, y - pad),
                       min(routing_matrix.shape[1], x + w + pad),
                       min(routing_matrix.shape[0], y + h + pad))

    # --- 2. SKELETONIZATION & PRE-PROCESSING ---
    # FIX 1: Upgraded Gap Bridger. 9x9 kernel to fuse larger pen skips (fixes W10/W26)
    bridged_matrix = cv2.morphologyEx(routing_matrix, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    skeleton = cv2.ximgproc.thinning(bridged_matrix, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skel_bin = (skeleton > 0).astype(np.uint8)

    # Label each separate wire with a unique ID number
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel_bin, connectivity=8)

    output_img = original_img.copy()

    print("\n" + "=" * 40)
    print("      DETECTED CIRCUIT TOPOLOGY")
    print("=" * 40)

    generated_netlist = []

    # --- 3. ROBUST CONNECTION DETECTION (Vector Logic) ---
    for g in gates:
        hx1, hy1, hx2, hy2 = g["hitbox"]
        hitbox_region = labels[hy1:hy2, hx1:hx2]

        touching_wires = np.unique(hitbox_region)
        touching_wires = touching_wires[touching_wires > 0]

        gate_center_y = g["y"] + (g["h"] // 2)
        gate_center_x = g["x"] + (g["w"] // 2)

        for wire_id in touching_wires:
            # Global Dust Filter
            if stats[wire_id, cv2.CC_STAT_AREA] < 15:
                continue

            wy, wx = np.where(hitbox_region == wire_id)

            # Penetration Filter
            if len(wy) < 4:
                continue

            global_wy = wy + hy1
            global_wx = wx + hx1

            mean_y = int(np.mean(global_wy))
            mean_x = int(np.mean(global_wx))

            # FIX 2: Vector-Based Pin Detection
            # Calculate distance from gate center to the wire connection point
            dy = mean_y - gate_center_y
            dx = mean_x - gate_center_x

            if abs(dy) >= abs(dx):
                # Wire is primarily on the Y-axis (Top or Bottom)
                pin_type = "INPUT" if dy < 0 else "OUTPUT"
            else:
                # Wire is primarily on the X-axis (Left or Right)
                pin_type = "INPUT" if dx < 0 else "OUTPUT"

            log_msg = f"{g['type']} Gate (ID:{g['id']}) has an {pin_type} connected to Wire #{wire_id}"
            print(log_msg)
            generated_netlist.append(log_msg)

            color = (0, 255, 0) if pin_type == "INPUT" else (0, 0, 255)
            cv2.circle(output_img, (mean_x, mean_y), 6, color, -1)
            cv2.putText(output_img, f"W{wire_id}", (mean_x + 10, mean_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- 4. VISUALIZATION ---
    for g in gates:
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(output_img, f"{g['type']}_{g['id']}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if np.max(labels) > 0:
        label_hue = np.uint8(179 * labels / np.max(labels))
    else:
        label_hue = np.uint8(labels)

    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    wire_mask = labels > 0
    output_img[wire_mask] = labeled_img[wire_mask]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Netlist Generation: Vector Math & Hardened Wires")
    plt.axis('off')
    plt.show(block = False)
    plt.pause(-1)

    return generated_netlist

