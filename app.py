import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request
from io import BytesIO
import requests

st.set_page_config(page_title="Face Swap - Central Face", layout="centered")
st.title("Face Swap with OpenCV LBF Landmarks (Central Face)")

# ---------------------------
# LBF Model
# ---------------------------
LBF_MODEL_URL = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBF_MODEL_FILE = "lbfmodel.yaml"

def ensure_lbf_model(path=LBF_MODEL_FILE, url=LBF_MODEL_URL):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    try:
        st.info(f"Downloading LBF model (~50MB)...")
        urllib.request.urlretrieve(url, path)
        st.success("Downloaded lbfmodel.yaml")
        return path
    except Exception as e:
        st.error(f"Failed to download LBF model: {e}")
        return None

# ---------------------------
# OpenCV Face Module
# ---------------------------
try:
    _ = cv2.face
except Exception:
    st.error(
        "OpenCV Face module not available.\n"
        "Install opencv-contrib-python:\n"
        "`pip install opencv-contrib-python`"
    )
    st.stop()

model_path = ensure_lbf_model()
if model_path is None:
    st.stop()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(model_path)

# ---------------------------
# Helpers
# ---------------------------
def select_central_face(faces, img_shape):
    """Select the face closest to the image center"""
    if len(faces) == 0:
        return None
    img_center = np.array([img_shape[1]//2, img_shape[0]//2])
    min_dist = float('inf')
    central_face = None
    for (x, y, w, h) in faces:
        face_center = np.array([x + w//2, y + h//2])
        dist = np.linalg.norm(face_center - img_center)
        if dist < min_dist:
            min_dist = dist
            central_face = np.array([[x, y, w, h]])
    return central_face

def get_landmarks(img_gray, faces):
    try:
        ok, landmarks = facemark.fit(img_gray, faces)
    except Exception:
        try:
            ok, landmarks = facemark.fit(img_gray, list(faces))
        except Exception as e2:
            st.error(f"facemark.fit failed: {e2}")
            return False, None
    return bool(ok), landmarks

def extract_points(landmarks):
    if landmarks is None:
        raise ValueError("landmarks is None")
    if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0:
        lm = np.asarray(landmarks[0])
        if lm.ndim == 3:
            lm = lm.reshape(-1, 2)
        pts = [(int(float(x)), int(float(y))) for x, y in lm]
        return np.array(pts, dtype=np.int32)
    if isinstance(landmarks, np.ndarray):
        lm = landmarks
        if lm.ndim == 3 and lm.shape[0] == 1:
            lm = lm.reshape(-1, 2)
        pts = [(int(float(x)), int(float(y))) for x, y in lm]
        return np.array(pts, dtype=np.int32)
    raise ValueError(f"Unknown landmark type: {type(landmarks)}")

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if img1_rect.size == 0:
        return

    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    img2_slice = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_slice * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] += img2_rect

# ---------------------------
# Face Swap Function
# ---------------------------
def face_swap(base_img, selfie_img):
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    selfie_gray = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2GRAY)

    # Detect central face only
    faces_base = select_central_face(face_cascade.detectMultiScale(base_gray, 1.1, 5), base_img.shape)
    faces_selfie = select_central_face(face_cascade.detectMultiScale(selfie_gray, 1.1, 5), selfie_img.shape)

    if faces_base is None or faces_selfie is None:
        st.error("Could not detect central face in one or both images.")
        return None

    ok_b, landmarks_base = get_landmarks(base_gray, faces_base)
    ok_s, landmarks_selfie = get_landmarks(selfie_gray, faces_selfie)

    if not ok_b or not ok_s:
        st.error("Could not find facial landmarks.")
        return None

    points1 = extract_points(landmarks_selfie)
    points2 = extract_points(landmarks_base)

    if len(points1) < 68 or len(points2) < 68:
        st.warning(f"Landmark count: selfie={len(points1)}, base={len(points2)}")

    img1 = np.float32(selfie_img)
    img2 = np.float32(base_img.copy())

    hull_index = cv2.convexHull(points2, returnPoints=False)
    hull1 = [tuple(points1[int(idx)]) for idx in hull_index.flatten()]
    hull2 = [tuple(points2[int(idx)]) for idx in hull_index.flatten()]

    # Delaunay triangulation
    rect = cv2.boundingRect(np.array(hull2))
    subdiv = cv2.Subdiv2D(rect)
    for p in hull2:
        subdiv.insert(np.float32([p[0], p[1]]))

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idxs = []
        for pt in pts:
            for i in range(len(hull2)):
                if abs(pt[0]-hull2[i][0]) < 1 and abs(pt[1]-hull2[i][1]) < 1:
                    idxs.append(i)
        if len(idxs) == 3:
            indexes_triangles.append(idxs)

    for triangle_index in indexes_triangles:
        t1 = [hull1[triangle_index[0]], hull1[triangle_index[1]], hull1[triangle_index[2]]]
        t2 = [hull2[triangle_index[0]], hull2[triangle_index[1]], hull2[triangle_index[2]]]
        warp_triangle(img1, img2, t1, t2)

    # Seamless clone
    hull8U = np.array(hull2, dtype=np.int32)
    mask = np.zeros(base_img.shape, dtype=base_img.dtype)
    cv2.fillConvexPoly(mask, hull8U, (255, 255, 255))

    r = cv2.boundingRect(hull8U)
    center = (r[0]+r[2]//2, r[1]+r[3]//2)
    output = cv2.seamlessClone(np.uint8(img2), base_img, mask, center, cv2.NORMAL_CLONE)

    return output

# ---------------------------
# Streamlit UI
# ---------------------------
st.write("Upload images **or provide URLs**. Only central face will be swapped.")

def load_image(input_source):
    try:
        if isinstance(input_source, str) and input_source.startswith("http"):
            response = requests.get(input_source)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(input_source).convert("RGB")
        return np.array(img)[:, :, ::-1]  # RGB to BGR
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None

base_file = st.file_uploader("Base Image (target)", type=["jpg","jpeg","png","webp"])
selfie_file = st.file_uploader("Selfie Image (source)", type=["jpg","jpeg","png","webp"])
base_url = st.text_input("Or enter Base Image URL")
selfie_url = st.text_input("Or enter Selfie Image URL")

if st.button("Swap Faces"):
    base_img = load_image(base_file if base_file else base_url)
    selfie_img = load_image(selfie_file if selfie_file else selfie_url)

    if base_img is not None and selfie_img is not None:
        with st.spinner("Swapping faces..."):
            result = face_swap(base_img, selfie_img)
        if result is not None:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Face Swap Result", use_column_width=True)
        else:
            st.warning("Face swap failed. Make sure central faces are visible and frontal.")
