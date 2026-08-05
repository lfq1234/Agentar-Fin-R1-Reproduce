import asyncio, os, tempfile, sys
sys.path.insert(0, os.path.dirname(__file__))
from app.db.history import store as storemod

async def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "history.db")
    s = storemod.SessionHistoryStore(db, record_traces=True, fail_mode="silent", admin_user_ids=["admin1"])
    print("connect", flush=True)
    await s.connect()
    print("init", flush=True)
    await s.init_tables()
    print("plain lock reacquire test:", flush=True)
    async with s._lock:
        print("  lock1 ok", flush=True)
    async with s._lock:
        print("  lock2 ok", flush=True)
    print("lock fine", flush=True)
    await s.close()
    print("DONE", flush=True)

asyncio.run(asyncio.wait_for(main(), 25))
