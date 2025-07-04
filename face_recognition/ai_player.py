from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import random
import time
from face_recognition.authorize_face import FaceDB
import os
from dotenv import load_dotenv
load_dotenv()


class AIPlayer:
    @staticmethod
    def open_ytmusic_plyalist(playlist_id):
        options = Options()
        user_data_dir = r"C:\\selenium\\profile"
        options.add_argument(f"--user-data-dir={user_data_dir}")
        options.add_argument("--profile-directory=Default")
        options.add_argument("--start-minimized")
        options.add_experimental_option("detach", True)
        driver = webdriver.Chrome(options=options)

        driver.get(f"https://music.youtube.com/watch?list={playlist_id}")

        xpath = '// *[ @ id = "play-button"]'
        wait = WebDriverWait(driver, 10)
        wait.until(EC.visibility_of_element_located((By.XPATH, xpath))).click()

    @staticmethod
    def connect_to_speaker():
        import socket

        # Target Bluetooth device address and RFCOMM port
        bd_addr = os.getenv("target_mac_address")
        # dev_name = ""
        port = 1

        # Create a Bluetooth socket
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)

        try:
            sock.connect((bd_addr, port))
        except:
            pass
        finally:
            time.sleep(5)
            # sock.close()

    @classmethod
    def play_user_plyalist_using_faceid(cls):
        fdb = FaceDB()
        user_id, username = fdb.authorize_user_cam()

        playlists = fdb.get_playlists_by_userid(user_id)
        playlist_id = random.choice(playlists)
        cls.connect_to_speaker()
        cls.open_ytmusic_plyalist(playlist_id[0])


if __name__ == "__main__":
    AIPlayer.play_user_plyalist_using_faceid()
