# Phononic Nanobeam Design Lab

**Research owner: Athul Prem | `_AP_pnc`**

可配置的三维各向异性弹性波研究工具，面向高纯锗纳米梁。当前版本已实现真实三维直梁的本征振动、周期导波色散、可选圆孔几何与离线交互结果页。采用 Python / scikit-fem / SciPy / Gmsh，当前求解后端运行于 CPU。

## 快速开始

Python 3.12；建议独立虚拟环境。在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_3d_AP_pnc.txt
.\.venv\Scripts\python.exe simulate_AP_pnc.py --open
```

已经配置好本地环境时，可以双击 `Launch_AP_pnc.cmd`。默认算例是长2000 nm、宽1000 nm、厚200 nm的均匀三维悬臂梁，左端固定，其余表面无牵引。它是小型数值基准，不是26孔器件的最终研究结果。

参数来自 [straight_beam_AP_pnc.json](configs/straight_beam_AP_pnc.json)，无需修改源码。例如：

```powershell
# 同一面内几何，改变真实厚度
.\.venv\Scripts\python.exe simulate_AP_pnc.py --thickness-nm 100 --open

# 无限均匀波导的一个周期单胞：两端Bloch约束、侧面自由
.\.venv\Scripts\python.exe simulate_AP_pnc.py --analysis bloch --cells 4 3 2 --k-points 9 --open

# 自由有限梁；请求数量包含6个刚体零模
.\.venv\Scripts\python.exe simulate_AP_pnc.py --boundary free --modes 10
```

每个算例在 `results/runs/` 下生成：可交互的 `report_AP_pnc.html`、模式CSV、解析后的配置、带源码与环境信息的摘要、保留完整二次单元自由度映射的NPZ，以及方便ParaView查看的VTU。VTU仅采样顶点、显示归一化振型；它不是实际激励下的位移幅度。Bloch表在每个波数按频率独立排序，尚未跨波数跟踪分支。

相同配置、源码与数值环境的完整结果通过文件校验后复用。失败和中断的尝试保留，重试另开目录。没有大规模频率任务的自动断点续算功能；逐频点安全恢复将在散射批处理阶段加入。

## 已验证与尚未实现

本次Windows/Python 3.12本地验收有32项测试，31项通过、1项符号链接权限测试跳过，无失败。验证了张量旋转、均匀应变、一致质量、刚体子空间、独立密集/稀疏对照、Bloch周期相位和自由侧面的长波杆极限。

[三层网格证据](verification/straight_beam_validation_AP_pnc.json)针对默认尺寸的前四个悬臂模式，使用1755、5187、11475自由度。最后两层最大频率变化约0.468%，最小质量加权振型重叠约0.999992；固定物性20 mK / 4 K独立组装与求解得到相同矩阵与频率。该结论限定于这个基准，不代表所有几何、厚度、频段都收敛。

| 能力 | 当前状态 |
| --- | --- |
| 真实三维均匀梁、P1/P2矢量四面体 | 可运行 |
| 名义立方晶体弹性、实验室坐标旋转 | 可运行；物性尚未按低温样品校准 |
| 有限梁自由/单端/双端夹持本征模式 | 可运行 |
| 自由侧壁波导Bloch色散 | 可运行；有限分支采样，不代表全模式完备 |
| 26孔输入与净距检查 | 已通过；支持旧position_nm与示例x_nm格式 |
| Gmsh三维通孔几何 | 两孔小案例已测；完整26孔计算尚未运行 |
| 端口功率透射、反射与模式转换 | 下一阶段；不可用位移比替代 |
| 温度相关物性、真实损耗、弯梁和环 | 后续研究 |
| GPU求解 | 尚未实现或测速；先依据CPU实测确定瓶颈 |

当前零应力线性弹性模型不能给出未经建模的本征损耗、辐射Q、实验性能或单声子量子态。完成一个求解不等于通过网格收敛或实验验证。离线结果页的脚本交互已检查；受本机浏览器访问策略限制，尚未完成真实浏览器截图验收。

## 运行验证

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test*_AP_pnc.py' -v
.\.venv\Scripts\python.exe -B tools/validate_3d_AP_pnc.py
```

验收工具使用独立的三网格计算、质量加权振型匹配和温度零效应对照。失败门槛按实际结果返回非零退出码，不插值填补或更改物性以通过。`--quick`只检验工具执行路径，粗网格可能不通过0.5%频率变化门槛。

## 开发路线与资料

- [当前实施方案与代码复用判断](docs/IMPLEMENTATION_PLAN_AP_pnc.md)
- [紧凑开发交接](docs/AI_HANDOFF_AP_pnc.md)
- [完整物理规格](docs/PHYSICS_SPECIFICATION_AP_pnc.md)
- [长期研究工作包](docs/NEXT_STEPS_AP_pnc.md)
- [材料参考来源与适用边界](docs/REFERENCES_AP_pnc.md)

当前优先级是匹配端口、按能流归一化与无孔直梁的单位功率透射基准，之后才把26孔器件接入功率计算。QTCAD用于参考清晰的工作流，未作为声子求解后端。

## 历史二维证据

旧文件保留在 `archive/`、`results/archive_*`、`figures/archive_*`、`provenance/`。当前Git下载约32 MB，不为节省少量空间删除唯一科学证据。它们不参与新求解器启动。

原上传199个文件的字节保存完整，但旧清单仍有223项未解决（222缺失、1项依赖文件不同）。最新完整二维流程无法精确重现；新三维实现独立开发，不以找齐旧文件为前提。历史近3GHz带隙是二维面内结果，约−42.25 dB是探针位移比，均不能直接作为三维功率结论。[详细证据与缺件](docs/EVIDENCE_AND_RECOVERY_AP_pnc.md)

旧入口继续用于保存数据检查，不启动新三维计算：

```powershell
.\.venv\Scripts\python.exe tools/audit_repository_AP_pnc.py
.\.venv\Scripts\python.exe verify_AP_pnc.py --fields --safety
.\.venv\Scripts\python.exe run_AP_pnc.py --name hpge_reference --dry-run
```

新图和项目文件保留 `Athul Prem | _AP_pnc` 来源标识。最终研究报告要求仍位于 `reports/`，实际对外发送须由当前用户授权。
