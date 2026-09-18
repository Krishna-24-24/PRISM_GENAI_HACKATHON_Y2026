def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_search_empty_query(client):
    response = client.post("/search?query=")
    assert response.status_code == 400
    assert "error" in response.json()
