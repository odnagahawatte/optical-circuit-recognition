import cv2
import numpy as np
import matplotlib.pyplot as plt


def classify_complex_circuit_v25_custom(image_path, show_output=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return []

    # --- 1. RESIZE & THRESHOLD (The Stable v25 Foundation) ---
    target_width = 800
    aspect = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (target_width, int(target_width * aspect)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8)

    binary[0:15, :] = 0;
    binary[-15:, :] = 0
    binary[:, 0:15] = 0;
    binary[:, -15:] = 0

    # --- 2. FLOOD FILL & ISOLATE ---
    thick = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    contours_for_fill, _ = cv2.findContours(thick, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_temp = thick.copy()
    for cnt in contours_for_fill:
        cv2.drawContours(filled_temp, [cnt], -1, 255, thickness=cv2.FILLED)

    filled_temp = cv2.morphologyEx(filled_temp, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    gates_only = cv2.morphologyEx(filled_temp, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(gates_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = img.copy()

    if show_output:
        # UPDATED: Added Aspect Ratio to the console table
        print(
            f"{'TYPE':<10} | {'SCALPEL':<7} | {'SOLIDITY':<8} | {'EXTENT':<8} | {'CIRCULARITY':<11} | {'ASPECT RATIO':<12}")
        print("-" * 75)

    total_image_area = img.shape[0] * img.shape[1]
    detected_components = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # The v25 Dynamic Noise Floor (0.1%)
        min_area = total_image_area * 0.001
        max_area = total_image_area * 0.3

        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)

            pad = 30
            y1, y2 = max(0, y - pad), min(filled_temp.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(filled_temp.shape[1], x + w + pad)
            roi = filled_temp[y1:y2, x1:x2]

            if roi.size == 0: continue

            k_base = int(min(w, h) * 0.15)
            k_size = k_base if k_base % 2 != 0 else k_base + 1
            k_size = max(13, k_size)

            clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            clean_roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, clean_kernel)

            roi_contours, _ = cv2.findContours(clean_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(roi_contours) > 0:
                body_cnt = max(roi_contours, key=cv2.contourArea)
                b_area = cv2.contourArea(body_cnt)

                if b_area < 50: continue

                hull = cv2.convexHull(body_cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(b_area) / hull_area if hull_area > 0 else 0

                bx, by, bw, bh = cv2.boundingRect(body_cnt)
                extent = float(b_area) / (bw * bh) if (bw * bh) > 0 else 0

                perimeter = cv2.arcLength(body_cnt, True)
                circularity = (4 * np.pi * b_area) / (perimeter ** 2) if perimeter > 0 else 0

                # UPDATED: Calculate Aspect Ratio
                aspect_ratio = float(bw) / bh if bh > 0 else 0

                # --- CUSTOM V25 HEURISTIC (With Aspect Ratio Shield) ---

                # Shield Added: NOT gates must have low extent AND be tall/square (Aspect Ratio < 1.15)
                if extent < 0.58 and aspect_ratio < 1.15:
                    gate_type = "NOT"
                    color = (255, 0, 255)
                elif solidity > 0.85 and circularity > 0.65:
                    gate_type = "AND"
                    color = (0, 255, 0)
                else:
                    gate_type = "OR"
                    color = (0, 165, 255)

                if show_output:
                    print(
                        f"{gate_type:<10} | {k_size}x{k_size:<5} | {solidity:<8.3f} | {extent:<8.3f} | {circularity:<11.3f} | {aspect_ratio:<12.3f}")

                global_x = x1 + bx
                global_y = y1 + by

                cv2.rectangle(output_img, (global_x, global_y), (global_x + bw, global_y + bh), color, 3)
                cv2.putText(output_img, f"{gate_type} ({solidity:.2f})", (global_x, global_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # STORE FOR MAIN.PY
                detected_components.append({
                    "type": gate_type,
                    "x": global_x,
                    "y": global_y,
                    "w": bw,
                    "h": bh
                })

    if show_output:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("1. Thresholded")
        plt.imshow(binary, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("2. Isolated Gates")
        plt.imshow(gates_only, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("3. Final Result")
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return detected_components


# Test it standalone:
if __name__ == "__main__":
    # Test on an image with a heavily horned OR gate to watch the aspect ratio shield work!
    classify_complex_circuit_v25_custom('new8.jpeg')