import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detection", page_icon="🕳️", layout="wide")
st.title("🕳️ Pothole Detection")
st.markdown("Using Traditional CV and Deep Learning (YOLOv8-Seg) with Depth Estimation.")

@st.cache_resource
def load_models():
    import os
    import gc
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    

    yolo = YOLO("runs/segment/train-4/weights/best.pt")
    gc.collect() 
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True).to("cpu")
    midas.load_state_dict(torch.load("midas_v21_small_256.pt", map_location="cpu", weights_only=True))
    midas.eval()
    
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    gc.collect() 
    
    return yolo, midas, transform

with st.spinner("⏳ Loading models..."):
    yolo_model, midas_model, midas_transform = load_models()

def calculate_stats(widths):
    if not widths:
        return 0, 0, 0, 0
    return len(widths), round(np.mean(widths), 2), max(widths), min(widths)

def display_summary(count, avg, mx, mn):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potholes Detected", f"{count}")
    c2.metric("Average Width", f"{avg} cm")
    c3.metric("Maximum Width", f"{mx} cm")
    c4.metric("Minimum Width", f"{mn} cm")

def get_dynamic_pixel_ratio(y_pos, image_height, ratio_top=0.3, ratio_bottom=0.08):
    y_norm = y_pos / image_height
    return ratio_top + (ratio_bottom - ratio_top) * y_norm

def get_depth_map(img_rgb, model, transform):
    input_batch = transform(img_rgb).to("cpu")
    with torch.no_grad():
        depth = model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(img_rgb.shape[0], img_rgb.shape[1]), 
            mode="bicubic", align_corners=False
        ).squeeze()
    depth_np = depth.cpu().numpy()
    return (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

def draw_visuals(img, x, w, y, h, width_cm):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    teks = f"{width_cm} cm"
    (text_w, text_h), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 5, y), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, teks, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

uploaded_file = st.file_uploader("📸 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    if isinstance(uploaded_file, list):
        st.error("Please upload only one image at a time.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_bgr.shape[:2]

        st.image(img_rgb, caption="Original Image", use_container_width=True)
        st.divider()
        
        with st.spinner("Analyzing..."):
            depth_map = get_depth_map(img_rgb, midas_model, midas_transform)
            f_length = 500
            z_scale = (50, 300)

            w_q1, w_q2, w_q3, w_q4 = [], [], [], []
            canvas_q1, canvas_q2, canvas_q3, canvas_q4 = [img_bgr.copy() for _ in range(4)]

            # --- TRADISIONAL CV ---
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
            canny = cv2.Canny(blurred, 60, 180)
            cleaned = cv2.morphologyEx(cv2.bitwise_or(adaptive, canny), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            merged_trad_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for cnt in contours:
                if cv2.contourArea(cnt) >= 500:
                    cv2.drawContours(merged_trad_mask, [cnt], 0, 255, -1)
            
            final_trad_contours, _ = cv2.findContours(merged_trad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in final_trad_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                y_max = min(y + h, img_h - 1)
                x_center = min(int(x + w / 2), img_w - 1)
                
                ratio = get_dynamic_pixel_ratio(y_max, img_h)
                val_q1 = round(w * ratio, 2)
                w_q1.append(val_q1)
                draw_visuals(canvas_q1, x, w, y, h, val_q1)

                Z = z_scale[0] + depth_map[y_max, x_center] * (z_scale[1] - z_scale[0])
                val_q2 = round((w * Z) / f_length, 2)
                w_q2.append(val_q2)
                draw_visuals(canvas_q2, x, w, y, h, val_q2)

            # --- YOLOv8-SEG ---
            results = yolo_model(img_rgb, verbose=False)[0]
            if results.masks is not None:
                mask_tensor = results.masks.data.cpu().numpy()
                mask_combined = np.max(mask_tensor, axis=0)
                yolo_mask = cv2.resize(mask_combined, (img_w, img_h))
                yolo_mask = (yolo_mask > 0).astype(np.uint8) * 255
                
                yolo_contours, _ = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in yolo_contours:
                    x, y, w, h_box = cv2.boundingRect(cnt)
                    y_max = min(y + h_box, img_h - 1)
                    x_center = min(int(x + w / 2), img_w - 1)
                    
                    ratio = get_dynamic_pixel_ratio(y_max, img_h)
                    val_q3 = round(w * ratio, 2)
                    w_q3.append(val_q3)
                    draw_visuals(canvas_q3, x, w, y, h_box, val_q3)

                    Z = z_scale[0] + depth_map[y_max, x_center] * (z_scale[1] - z_scale[0])
                    val_q4 = round((w * Z) / f_length, 2)
                    w_q4.append(val_q4)
                    draw_visuals(canvas_q4, x, w, y, h_box, val_q4)


        t_over, t_q1, t_q2, t_q3, t_q4 = st.tabs([
        "📊 Overview", 
        "🔍 Trad + Manual Y", 
        "🔍 Trad + MiDaS", 
        "🚀 YOLO + Manual Y", 
        "🚀 YOLO + MiDaS"
        ])
        
        with t_over:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### **Traditional CV + Manual Y-Axis**")
                st.image(cv2.cvtColor(canvas_q1, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown("#### **YOLO + Manual Y-Axis**")
                st.image(cv2.cvtColor(canvas_q3, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.markdown("#### **Traditional CV + MiDaS Depth**")
                st.image(cv2.cvtColor(canvas_q2, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.markdown("#### **YOLO + MiDaS Depth**")
                st.image(cv2.cvtColor(canvas_q4, cv2.COLOR_BGR2RGB), use_container_width=True)

        def fill_tab(tab, canvas, widths, title):
            with tab:
                st.markdown(f"### {title}")
                st.markdown("#### 📈 Stats")
                display_summary(*calculate_stats(widths))
                st.divider()
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

        fill_tab(t_q1, canvas_q1, w_q1, "Traditional CV + Manual Y-Axis")
        fill_tab(t_q2, canvas_q2, w_q2, "Traditional CV + MiDaS Depth")
        fill_tab(t_q3, canvas_q3, w_q3, "YOLO + Manual Y-Axis")
        fill_tab(t_q4, canvas_q4, w_q4, "YOLO + MiDaS Depth")
