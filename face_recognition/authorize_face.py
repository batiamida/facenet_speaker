import faiss
from dotenv import load_dotenv
import os
import sys
import sqlite3
from PIL import Image
import onnxruntime as ort
import cv2 as cv
from facenet_pytorch import MTCNN
import numpy as np
import mediapipe as mp
from face_recognition.pose_estimation import get_head_pose
from utils import FrameReciever
load_dotenv()


class FaceDB:
    root = os.getenv("ROOT_DIR")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH")
    model_path = os.path.join(root, "models", os.getenv("MODEL_NAME"))

    NEG_INF, POS_INF = -float("inf"), float("inf")
    DIRECTIONAL_THRESHOLDS = [((-10, 10), (-10, 10)), ((-30, 30), (NEG_INF, -10)),
                              ((NEG_INF, -15), (-30, 30)), ((15, POS_INF), (-30, 30))]
    TEXT_INSTRUCTIONS = ["look straight forward", "turn head up", "turn head left", "turn head right"]

    def __init__(self, conn_path: str):
        self.conn = sqlite3.connect(conn_path)
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                      thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, )
        self.ort_sess = None

    @staticmethod
    def create_index():
        vector_dimension = int(os.getenv("VECTOR_DIMENSION"))

        index_path = os.getenv("FAISS_INDEX_PATH")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatIP(vector_dimension)
        return index

    def save_face(self, img, name, user_exists=False):
        if self.ort_sess is None:
            self.ort_sess = ort.InferenceSession(self.model_path)
        res = self.ort_sess.run(["output"], {"input": np.array([img.numpy()])})
        norm_res = res[0]
        faiss.normalize_L2(norm_res)

        index = self.create_index()
        faiss_id = index.ntotal
        index.add(norm_res)
        faiss.write_index(index, self.faiss_index_path)
        cursor = self.conn.cursor()
        if not user_exists:
            cursor.execute("INSERT INTO user_info (username) VALUES (?)", (name,))
            user_id = cursor.lastrowid
        else:
            res = cursor.execute("SELECT user_id, username FROM user_info WHERE username = ?", (name,))
            user_id = res.fetchone()[0]

        cursor.execute("INSERT INTO face_info (user_id, faiss_id) VALUES (?, ?)", (user_id, faiss_id))
        self.conn.commit()
        cursor.close()
        return True

    def find_similar_faces(self, frame):
        img = Image.fromarray(frame)
        img = self.mtcnn(img)
        if img is not None:
            if self.ort_sess is None:
                self.ort_sess = ort.InferenceSession(self.model_path)
            res = self.ort_sess.run(["output"], {"input": np.array([img.numpy()])})[0]
            faiss.normalize_L2(res)
            index = self.create_index()
            distances, indices = index.search(res, k=1)
            return distances, indices
        return None, None

    def add_new_yt_playlist(self, user_id, playlist_id, **kwargs):
        cur = self.conn.cursor()
        other_cols = [kwargs.get(col) for col in ["playlist_name", "playlist_genre",
                                                  "playlist_mood", "playlist_description"]]
        cur.execute(f"""
                    INSERT INTO user_playlists (user_id, playlist_id, playlist_name,
                                                playlist_genre, playlist_mood, playlist_description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_id, playlist_id, *other_cols))
        self.conn.commit()
        cur.close()
        return True

    def add_new_identity_using_cam(self, name):
        mp_face_mesh = mp.solutions.face_mesh
        current_state = 0
        cam = self.get_cam()
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
            while True:
                frame, frame_arr, image = self.get_preprocessed_frame(cam, current_state)

                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    cv.putText(frame, "processing...", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    landmarks = results.multi_face_landmarks[0].landmark
                    yaw, pitch = get_head_pose(landmarks, frame.shape)

                    thresh1, thresh2 = self.DIRECTIONAL_THRESHOLDS[current_state]
                    head_pos_correct = lambda yaw, pitch: \
                        (thresh1[0] <= yaw <= thresh1[1]) and (thresh2[0] <= pitch <= thresh2[1])
                    if head_pos_correct(yaw, pitch):
                        img = self.mtcnn(frame_arr)
                        if img is not None:
                            face_saved = self.save_face(img, name, current_state > 0)
                            current_state += face_saved

                            if current_state == len(self.DIRECTIONAL_THRESHOLDS):
                                break

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            cam.release()
            cv.destroyAllWindows()
            return True

    def get_preprocessed_frame(self, cam, current_state):
        ret, frame = cam.read()
        frame = cv.flip(frame, 1)
        frame_arr = Image.fromarray(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.putText(frame, f"follow the isntruction: {self.TEXT_INSTRUCTIONS[current_state]}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Pose estimation", frame)
        return frame, frame_arr, image

    def authorize_user_cam(self):
        while True:
            cam = self.get_cam()
            ret, frame = cam.read()
            distances, indices = self.find_similar_faces(frame)
            if distances is not None:
                if distances[0][0] >= 0.6:
                    cam.release()
                    cv.destroyAllWindows()

                    idx = indices[0][0]
                    user_id, username = self.get_user_info_by_faceid(idx)
                    return user_id, username

    def get_user_info_by_faceid(self, idx):
        cur = self.conn.cursor()
        res = cur.execute(f"""
            SELECT user_info.user_id, user_info.username FROM user_info
            JOIN face_info as finfo ON finfo.user_id == user_info.user_id
            WHERE finfo.faiss_id == {idx}
        """).fetchone()
        cur.close()

        return res

    def get_playlists_by_userid(self, user_id):
        cur = self.conn.cursor()
        res = cur.execute(f"""
            SELECT playlist_id, playlist_name,
                    playlist_genre, playlist_mood, playlist_description
            FROM user_playlists
            WHERE user_id == {user_id}               
        """).fetchall()
        cur.close()
        res = [dict(playlist_id=val1, playlist_name=val2,
                    playlist_genre=val3, playlist_mood=val4,
                    playlist_description=val5) for val1, val2, val3, val4, val5 in res]
        return res

    def get_cam(self):
        try:
            cam = cv.VideoCapture(0)
        except:
            cam = iter(FrameReciever(os.getenv("camera_link"), os.getenv("camera_auth_token")))

        return cam


if __name__ == "__main__":
    name = "Someone"
    fdb = FaceDB(r'your_path_to_db')
    fdb.add_new_identity_using_cam(name)

    # user_id = ...
    # youtube_playlist_id = ...
    # fdb.add_new_yt_playlist(user_id, youtube_playlist_id,
    #                         playlist_name="",
    #                         playlist_genre="",
    #                         playlist_mood="",
    #                         playlist_description="")