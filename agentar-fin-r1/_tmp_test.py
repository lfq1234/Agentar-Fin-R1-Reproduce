import sys, types, json

# ---- stub 重依赖 ----
class _Stub(types.ModuleType):
    def __getattr__(self, n):
        return lambda *a, **k: None
sys.modules["pandas"] = _Stub("pandas")
sys.modules["pyarrow"] = _Stub("pyarrow")

torch = types.ModuleType("torch")
torch.zeros_like = lambda *a, **k: None
sys.modules["torch"] = torch

verl = types.ModuleType("verl"); verl.DataProto = object
workers = types.ModuleType("verl.workers")
rm = types.ModuleType("verl.workers.reward_manager")
naive = types.ModuleType("verl.workers.reward_manager.naive")
def register(name):
    def deco(cls):
        return cls
    return deco
rm.register = register
class NaiveRewardManager:
    pass
naive.NaiveRewardManager = NaiveRewardManager
verl.workers = workers
workers.reward_manager = rm
rm.naive = naive
sys.modules.update({"verl": verl, "verl.workers": workers,
                    "verl.workers.reward_manager": rm,
                    "verl.workers.reward_manager.naive": naive})
openai = types.ModuleType("openai"); openai.OpenAI = object
sys.modules["openai"] = openai

import importlib.util
spec = importlib.util.spec_from_file_location("pvd", "training/data/prepare_verl_data.py")
pvd = importlib.util.module_from_spec(spec); spec.loader.exec_module(pvd)
spec2 = importlib.util.spec_from_file_location("fjr", "training/grpo/fin_judge_reward.py")
fjr = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fjr)

# 真实样本（主格式：messages 数组，HUMAN/ASSISTANT，答案在 \\boxed{}）
sample = [
    {"role": "HUMAN", "content": "Question: what is the net change in accumulated other comprehensive income during 2015?\nAnswer:"},
    {"role": "ASSISTANT", "content": "<think>... The net change is 60 - 666 = -606 million. **Answer:** \\boxed{-606}"},
]
ex = pvd._extract_messages_from_row(sample)
print("role normalize:", [m["role"] for m in ex["sft_messages"]])
print("gold answer   :", repr(ex["answer"]))
print("verifiable    :", pvd._is_verifiable(ex["answer"]))
gt = json.dumps({"answer": ex["answer"], "verifiable": pvd._is_verifiable(ex["answer"])}, ensure_ascii=False)
print("ground_truth  :", gt)

assistant = ex["sft_messages"][2]["content"]
print("\n[RLVR] 正确回答 ->", fjr._rlvr(assistant, {"answer": "-606", "verifiable": True}), "(应为 1.0)")
print("[RLVR] 错误回答 ->", fjr._rlvr("<think>wrong</think> \\boxed{100}", {"answer": "-606", "verifiable": True}), "(应为 0.0)")
print("[RLVR] 缺<think> ->", fjr._rlvr("\\boxed{-606}", {"answer": "-606", "verifiable": True}), "(应为 0.0)")
print("[RLVR] 开放题   ->", fjr._rlvr("<think>分析</think>\\boxed{略}", {"answer": "见上文", "verifiable": False}), "(应为 None)")

long_ans = "公司应综合考虑流动性风险和信用评级变化，建议采用阶梯式对冲策略以平滑损益波动。"
print("长文本 verifiable:", pvd._is_verifiable(long_ans), "(应为 False)")
legacy = {"messages": [{"role": "system", "content": "S"},
                       {"role": "user", "content": "U"},
                       {"role": "assistant", "content": "<think>t</think> \\boxed{42}"}]}
ex2 = pvd._extract_messages_from_row(legacy)
print("legacy gold:", repr(ex2["answer"]), "verif:", pvd._is_verifiable(ex2["answer"]))
print("\nALL CHECKS DONE")
