import asyncio, os, tempfile, sys
sys.path.insert(0, os.path.dirname(__file__))
from app.db.history import collect, store as storemod

class FakeResult:
    def __init__(self, reply="", compliance_notes=None, risk_flags=None):
        self.reply = reply
        self.compliance_notes = compliance_notes or []
        self.risk_flags = risk_flags or []

async def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "history.db")
    s = storemod.SessionHistoryStore(db, record_traces=True, fail_mode="silent", admin_user_ids=["admin1"])
    print("connect...", flush=True)
    await s.connect()
    print("init_tables...", flush=True)
    await s.init_tables()
    print("record_run...", flush=True)
    result = FakeResult(reply="答案是42", compliance_notes=["合规OK"], risk_flags=[])
    events = collect.build_events(result, user_message="什么是GDP")
    await s.record_run(
        conversation_id="conv-1", user_id="userA", scene="qa", run_id="run-conv-1",
        turn_id="turn-conv-1", duration_ms=123, model="m", result=result, events=events,
        user_message="什么是GDP", total_tokens=10)
    print("get_session...", flush=True)
    d = await s.get_session("userA", "conv-1")
    print("get_session OK:", d is not None, flush=True)
    await s.close()
    print("DONE", flush=True)

asyncio.run(asyncio.wait_for(main(), 20))
