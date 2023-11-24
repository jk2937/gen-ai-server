import sqlite3

class database:

    def __init__(self, dbname="database.db"):

        self.connection = sqlite3.connect(dbname)


    def create_table(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fish_inventory (
                id INTEGER PRIMARY KEY,
                name TEXT,
                species TEXT,
                quantity INTEGER
            )
            """
        )

        
        self.connection.commit()
        cursor.close()


    def insert_data(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO fish_inventory (name, species, quantity)
            VALUES (?, ?, ?)
            """, 
            ("Nemo", "Clownfish", 10)
        )


        self.connection.commit()
        cursor.close()


    def read_data(self):

        cursor = self.connection.cursor()

        cursor.execute("SELECT * FROM fish_inventory")
        fish_data = cursor.fetchall()

        for fish in fish_data:
            print(f"Fish ID: {fish[0]}, Name: {fish[1]}, Species: {fish[2]}, Quantity: {fish[3]}")


        cursor.close()


    def deinit(self):

        self.connection.close()


    def test(self):

        try:
            self.__init__()
        except ProgrammingError:
            print("Error")

        self.create_table()
        self.insert_data()
        self.read_data()


        self.deinit()


def main():

    db = database()
    db.test()


if __name__ == "__main__":
    
    main()
