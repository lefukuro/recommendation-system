import model_conn
from fastapi.testclient import TestClient 
from datetime import datetime


client = TestClient(model_conn.app)

user_id =  200     # enter the user ID (>199) and date 
time = datetime(2021, 12, 20)

try:
    r = client.get(
        f"/post/recommendations/",
        params={"id": user_id, "time": time, "limit": 5}
    )
except Exception as e:
    raise ValueError(f"X ошибка при выполнении запроса {type(e)} {str(e)}")

print(r.json())