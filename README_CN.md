# RankPSM 中文说明

这是论文 **RankPSM: Source-Calibrated Cross-Corpus Evaluation of Password Strength Meters** 的安全开源代码包。

## 这个仓库可以上传什么

可以上传：

- `src/`
- `scripts/`
- `run_*.py`
- `bootstrap_significance_v3_1.py`
- `requirements.txt`
- `results/` 中的聚合结果
- `paper_exports/` 中的论文表格
- `figures/` 中的图表
- `README.md`
- `README_CN.md`
- `DATA_SAFETY.md`
- `.gitignore`
- `LICENSE`

## 不能上传什么

绝对不要上传：

- `Password-Strength-Meter-Accuracy-master.zip`
- `data/psma_raw/`
- `data/real/`
- `prediction_audit_private_v3_1/`
- `.venv/`
- `.idea/`
- 任何原始密码文件
- 任何包含用户名、邮箱、IP、账号的文件

## 本地复现实验

先安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

用你本地下载的 PSMA benchmark 生成实验 CSV：

```powershell
python scripts/build_psma_result_csvs.py --raw .\data\psma_raw\Password-Strength-Meter-Accuracy-master\src --out .\data\real
python scripts/verify_no_identity_columns.py --data-root .\data\real
```

跑完整实验：

```powershell
python run_rankpsm_full_outputs_v3_1.py --data-root .\data\real
python bootstrap_significance_v3_1.py --B 1000
```
