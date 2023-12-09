import sqlite3
import datetime

class database:
    def __init__(self, dbname="database.db"):
        self.connection = sqlite3.connect(dbname)

        cursor = self.connection.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS images ( session_id TEXT, image_data BLOB, created_timestamp TEXT );")
        cursor.execute("CREATE TABLE IF NOT EXISTS jobs ( session_id TEXT, prompt TEXT, start_timestamp TEXT, status_message TEXT, current_count_index INT, count_max INT );")
        cursor.execute("CREATE TABLE IF NOT EXISTS nicknames ( session_id, nickname );")

        self.connection.commit()
        cursor.close()

    def add_image(self, session_id, image_data, created_timestamp):
        cursor = self.connection.cursor()

        cursor.execute(
            '''

            INSERT INTO images ( session_id, image_data, created_timestamp ) VALUES ( ?, ?, ? );

            ''',
            (session_id, image_data, created_timestamp)
        )

        self.connection.commit()
        cursor.close()

    def check_date_expired(self, date):
        return False

    def delete_old_images(self):
        cursor = self.connection.cursor()

        cursor.execute("SELECT created_timestamp FROM images")
        data = cursor.fetchall()

        for item in data:
            item = item[0]
            if self.check_date_expired(item):
                cursor.execute(f"DELETE FROM images WHERE created_timestamp = '{item}';")

        self.connection.commit()
        cursor.close()

    def delete_all_user_images(self, session_id):
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM images WHERE session_id = '{session_id}';")
        self.connection.commit()
        cursor.close()

    def create_job(self, session_id, prompt, count_max):
        start_timestamp = str(datetime.datetime.now().strftime("%H%M%S%f"))
        status_message = "Job created"
        cursor = self.connection.cursor()
        cursor.execute(f"INSERT INTO jobs ( session_id, prompt, start_timestamp, status_message, current_count_index, count_max) VALUES ( '{session_id}', '{prompt}', '{start_timestamp}', '{status_message}', 0, {count_max} );")
        self.connection.commit()
        cursor.close()

    def update_job_status_message(self, session_id, status_message):
        cursor = self.connection.cursor()
        cursor.execute(f"UPDATE jobs SET status_message = '{status_message}' WHERE session_id = '{session_id}';")
        
        self.connection.commit()
        cursor.close()

    def update_job_current_count_index(self, session_id, current_count_index):
        cursor = self.connection.cursor()
        cursor.execute(f"UPDATE jobs SET current_count_index = '{current_count_index}' WHERE session_id ='{session_id}';")
        
        self.connection.commit()
        cursor.close()

    def delete_job(self, session_id):
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM jobs WHERE session_id = '{session_id}';")
        self.connection.commit()
        cursor.close()

    def get_nickname(self, session_id):
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM nicknames WHERE session_id = '{session_id}';")
        nickname = cursor.fetchall()[0][1]

        self.connection.commit()
        cursor.close()
        return nickname

    def set_nickname(self, session_id, nickname):
        cursor = self.connection.cursor()
        cursor.execute(f"INSERT INTO nicknames ( session_id, nickname ) VALUES ( '{session_id}', '{nickname}' )")
        self.connection.commit()
        cursor.close()

    def update_nickname(self, session_id, nickname):
        cursor = self.connection.cursor()
        cursor.execute(f"UPDATE nicknames SET nickname = '{nickname}' WHERE session_id = '{session_id}';")
        self.connection.commit()
        cursor.close()

    def delete_nickname(self, session_id):
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM nicknames WHERE session_id = '{session_id}';")
        self.connection.commit()
        cursor.close()

    def deinit(self):
        self.connection.close()
