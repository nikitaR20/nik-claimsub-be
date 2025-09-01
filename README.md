Got it 👍 — I cleaned up your README so it’s **super beginner friendly**, fixes structure issues, and makes database + CSV loading steps crystal clear. Here’s the updated version:

---

# 🏥 Medical Insurance Claim Backend – Beginner Friendly

This backend project powers the **medical insurance claim app**.
It stores claim data, manages providers, customer segments, risk ratings, and allows the React frontend to fetch and submit claims.

> ⚠️ **Important:** The frontend will only work if the backend is running **and** connected to a PostgreSQL database.

---

## ✅ What You Need Before Starting

1. **A computer** (Windows, Mac, or Linux).
2. **Internet connection**.
3. **Visual Studio Code (recommended)** → [Download here](https://code.visualstudio.com/).
4. **Git** → [Download here](https://git-scm.com/downloads).
5. **Python 3.10 or higher** → [Download here](https://www.python.org/downloads/).
6. **PostgreSQL (with pgAdmin)** → [Download here](https://www.postgresql.org/download/).

---

## 🛠 Step 1: Install PostgreSQL

1. Download PostgreSQL from [here](https://www.postgresql.org/download/).
2. During installation:

   * Set a **password** for the `postgres` user → Remember this!
   * Install **pgAdmin**, a GUI to manage your database.
3. Open **pgAdmin** after installation.

---

## 🛠 Step 2: Create the Database

1. Open **pgAdmin** → log in with the password you created.
2. Right-click **Databases → Create → Database**.
3. Name the database:

   ```
   claims_db
   ```
4. Click **Save**.

---

## 🛠 Step 3: Get the Backend Project

1. Open **Visual Studio Code**.
2. Open the **Terminal** → (menu: `View → Terminal`).
3. Clone the backend repo:

```bash
git clone <project-repo-url>
```

4. Open the downloaded folder `backend_project` in Visual Studio Code.

---

## 🛠 Step 4: Create Virtual Environment & Install Dependencies

1. In the VS Code terminal, go inside your project folder:

```bash
cd backend_project
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it:

* **Windows:**

  ```bash
  venv\Scripts\activate
  ```
* **Mac/Linux:**

  ```bash
  source venv/bin/activate
  ```

4. Install required packages:

```bash
pip install -r requirements.txt
```

✅ Now your backend has all the Python libraries it needs.

---

## 🛠 Step 5: Create `.env` File

1. In the **`backend_project`** folder, create a new file named `.env`.
2. Add this inside:

```ini
DATABASE_URL=postgresql://postgres:your_postgres_password@localhost:5432/claims_db
SECRET_KEY=your_secret_key_here
```

* Replace `your_postgres_password` with the password you used when installing PostgreSQL.
* Replace `your_secret_key_here` with any random text (used for security).

---

## 🛠 Step 6: Load CSV Data into Database

Perfect 👍 Thanks for clarifying. Based on your steps, here’s a **super beginner-friendly README section** for **loading CSV data into PostgreSQL using pgAdmin**.

---

# 📥 Loading Data into PostgreSQL from CSV Files

Follow these steps carefully to load your project data into PostgreSQL:

---

## 1. Prepare Your CSV Files

* Collect all your CSV files (for example: `providers.csv`, `risk_ratings.csv`, etc.).
* Store them all inside **one folder** on your computer (e.g., `C:\Projects\backend_project\data`).

---

## 2. Update the SQL Files

* Inside the **`sql` folder** of this project, you will see multiple `.sql` files (e.g., `load_providers.sql`, `load_claims.sql`, etc.).
* Open each `.sql` file in **Visual Studio**.
* Modify the **file path** inside the SQL command to point to the correct CSV file location.
  Example for `providers`:

  ```sql
  COPY providers (provider_id, name, state, type)
  FROM 'C:/Projects/backend_project/data/providers.csv'
  DELIMITER ','
  CSV HEADER;
  ```

  🔴 **Important:** Use forward slashes `/` in file paths even on Windows (Postgres requires it).

---

## 3. Open pgAdmin and PSQL Tool

1. Open **pgAdmin** and log in with your database credentials.
2. Select your database (the one created earlier for this project).
3. Right-click → Choose **PSQL Tool** to open the SQL command workspace.

---

## 4. Run the SQL Scripts in Sequence

* In **Visual Studio**, open the SQL files one by one in the following order:

  1. `load_providers.sql`
  2. `load_risk_ratings.sql`
  3. `load_claims.sql`
  4. `load_claim_riders.sql`
  5. `load_claim_claim_rider.sql`
  6. `load_customer_segments.sql`
  7. `load_provider_customer_segments.sql`

* Copy the entire content of each file.

* Paste it into the **PSQL tool in pgAdmin**.

* Press **Enter** to execute.

✅ If everything runs successfully, your tables will now be filled with data from the CSV files.


## ▶️ Step 7: Run the Backend

1. Make sure your virtual environment is active (`venv` should show in terminal).Type pipenv shell to confirm.
2. Run this from inside `backend_project`(type in the terminal):

  uvicorn app.main:app --reload

3. If successful, backend runs at:

```
http://localhost:8000
```

Keep this terminal open while working.

---

## 📑 Step 8: Connect Frontend to Backend

1. Open your **frontend project folder**.
2. In the `.env` file, add:

```ini
REACT_APP_API_URL=http://localhost:5000
```

3. Now the frontend will talk to the backend.

---

## ❓ Troubleshooting

* **Database error:** Check if PostgreSQL is running and `.env` values are correct.
* **Backend won’t start:** Make sure you ran `pip install -r requirements.txt` and are using `python -m app.main`.
* **ModuleNotFoundError: No module named 'app'** → Run only from inside `backend_project`.
* **Port already in use:** Close other programs using port `5000` or change port in `.env`.

---

## 🎉 Done!
