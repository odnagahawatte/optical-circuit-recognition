import cv2
import numpy as np

# Import your Phase 1, Phase 2, and Phase 4 modules
from vision_engine import classify_complex_circuit_v25_custom
import wire_tracer
import simulator


def generate_routing_matrix(image_path, detected_gates):
    """
    Rebuilds the strict C=8 binary threshold to keep the background clean,
    then mathematically erases the gates to leave a pure wire map.
    """
    img = cv2.imread(image_path)
    target_width = 800
    aspect = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (target_width, int(target_width * aspect)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)

    # Strict C=8 Firewall to destroy paper texture and micro-shadows
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8)

    binary[0:15, :] = 0;
    binary[-15:, :] = 0
    binary[:, 0:15] = 0;
    binary[:, -15:] = 0

    # Erase the gates using the coordinates from the Vision Engine
    routing_matrix = binary.copy()
    for gate in detected_gates:
        x, y, w, h = gate['x'], gate['y'], gate['w'], gate['h']

        # Expand the erasure box slightly (padding) to ensure we snip the wires clean
        pad = 8
        cv2.rectangle(routing_matrix, (x - pad, y - pad), (x + w + pad, y + h + pad), 0, -1)

    return routing_matrix, img


def main():
    # Set your target image here
    target_image = 'new7.jpeg'

    print("[*] STEP 1: Booting Vision Engine...")
    # Run the vision engine silently to extract coordinates
    gates = classify_complex_circuit_v25_custom(target_image, show_output=False)

    if not gates:
        print("[-] No gates detected or image not found. Exiting.")
        return

    print(f"    -> Successfully extracted {len(gates)} logic gates:")
    for i, g in enumerate(gates):
        print(f"       Gate {i}: {g['type']} at ({g['x']}, {g['y']})")

    print("\n[*] STEP 2: Generating Routing Matrix...")
    routing_matrix, original_img = generate_routing_matrix(target_image, gates)
    print("    -> Wires isolated and background sterilized.")

    print("\n[*] STEP 3: Executing Wire Tracer...")
    # Hand off the gates and the pure wire map to your routing plugin
    final_netlist = wire_tracer.build_netlist(gates, routing_matrix, original_img)

    if not final_netlist:
        print("[-] No connections detected. Cannot run simulation.")
        return

    print("\n[*] STEP 4: Compiling Truth Table...")
    # Pass the generated netlist strings directly into the logic simulator
    simulator.evaluate_netlist(final_netlist)


if __name__ == "__main__":
    main()